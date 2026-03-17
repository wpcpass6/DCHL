# coding=utf-8
"""
THGR 训练脚本。

整体流程：
1) 读取参数并初始化随机种子；
2) 构建 TPOIDataset（内部会构建/加载静态异构超图缓存）；
3) 初始化 THGR 模型（图编码器 + 因果序列编码器 + 动态空间打分）；
4) 以 CrossEntropy 训练 next-POI 主任务，并叠加 mask fill 自监督损失；
5) 每个 epoch 在测试集评估 Recall/NDCG，按 Recall@5 保存最优模型。
"""

import argparse
import datetime
import logging
import os
import random
import time
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from metrics import batch_performance
from T_dataset import TPOIDataset, t_collate_fn
from T_model import THGR


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="TKY", help="NYC/TKY")
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--decay", type=float, default=5e-4)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--deviceID", type=int, default=0)

# THGR 图构建参数（静态异构超图）
parser.add_argument("--region_grid_size", type=float, default=0.02)
parser.add_argument("--cat_grid_size", type=float, default=0.01)
parser.add_argument("--geo_k", type=int, default=20)
parser.add_argument("--knn_chunk_size", type=int, default=512)
parser.add_argument("--num_hg_layers", type=int, default=2)
parser.add_argument("--poi_cat_filename", type=str, default="", help="可选：POI类别pkl文件（dict或list）")

# 序列编码参数（Transformer）
parser.add_argument("--n_tf_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--max_seq_len", type=int, default=256)

# 候选打分融合参数
parser.add_argument("--alpha_cat", type=float, default=0.2)
parser.add_argument("--beta_region", type=float, default=0.2)

# 自监督掩码恢复损失参数
parser.add_argument("--mask_ratio", type=float, default=0.05)
parser.add_argument("--lambda_mask", type=float, default=0.1)

parser.add_argument("--save_dir", type=str, default="logs")
args = parser.parse_args()


# 固定随机种子，确保可复现
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:{}".format(args.deviceID) if torch.cuda.is_available() else "cpu")

# 创建日志目录：logs/T_时间戳
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
current_save_dir = os.path.join(args.save_dir, "T_" + current_time)
os.mkdir(current_save_dir)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=os.path.join(current_save_dir, "log_training.txt"),
    filemode="w+",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

args_filename = args.dataset + "_T_args.yaml"
with open(os.path.join(current_save_dir, args_filename), "w") as f:
    yaml.dump(vars(args), f, sort_keys=False)


def main():
    """训练主入口。"""
    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info("device: %s", device)

    if args.dataset == "TKY":
        num_users = 2173
        num_pois = 7038
        padding_idx = num_pois
    elif args.dataset == "NYC":
        num_users = 834
        num_pois = 3835
        padding_idx = num_pois
    else:
        raise ValueError("Only NYC/TKY are supported in T_run.py")

    # 2) 构建训练/测试数据集。
    # 注意：TPOIDataset 在初始化时会优先读取缓存图，若无缓存再构图并写回。
    logging.info("2. Load Dataset")
    train_dataset = TPOIDataset(
        data_filename="datasets/{}/train_poi_zero.txt".format(args.dataset),
        pois_coos_filename="datasets/{}/{}_pois_coos_poi_zero.pkl".format(args.dataset, args.dataset),
        num_users=num_users,
        num_pois=num_pois,
        padding_idx=padding_idx,
        args=args,
        device=device,
    )
    test_dataset = TPOIDataset(
        data_filename="datasets/{}/test_poi_zero.txt".format(args.dataset),
        pois_coos_filename="datasets/{}/{}_pois_coos_poi_zero.pkl".format(args.dataset, args.dataset),
        num_users=num_users,
        num_pois=num_pois,
        padding_idx=padding_idx,
        args=args,
        device=device,
    )

    # 3) DataLoader 使用自定义 collate，对变长轨迹做 padding。
    logging.info("3. Construct DataLoader")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: t_collate_fn(b, padding_value=padding_idx, max_seq_len=args.max_seq_len),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: t_collate_fn(b, padding_value=padding_idx, max_seq_len=args.max_seq_len),
    )

    # 4) 初始化模型。
    # num_cats/num_regions 来自数据集实际构图结果，而不是手写常量。
    logging.info("4. Load Model")
    model = THGR(
        num_users=num_users,
        num_pois=num_pois,
        num_cats=train_dataset.num_cats,
        num_regions=train_dataset.num_regions,
        args=args,
        device=device,
    ).to(device)

    # 主任务使用多分类交叉熵：预测“下一 POI id”
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss().to(device)

    ks_list = [1, 5, 10, 20]
    best_rec5 = 0.0

    logging.info("5. Start Training")
    for epoch in range(args.num_epochs):
        # -------- 训练阶段 --------
        logging.info("================= Epoch %d/%d =================", epoch, args.num_epochs)
        start_time = time.time()

        model.train()
        train_loss = 0.0
        train_recall = np.zeros((len(train_loader), len(ks_list)), dtype=np.float32)
        train_ndcg = np.zeros((len(train_loader), len(ks_list)), dtype=np.float32)

        for i, batch in enumerate(train_loader):
            # 1) 前向：返回候选打分和 mask loss
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            pred, loss_mask = model(train_dataset, batch)
            # 2) 总损失 = next-POI 主损失 + lambda * mask 自监督
            loss_next = criterion(pred, batch["label"])
            loss = loss_next + args.lambda_mask * loss_mask

            # 3) 反向更新
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            for k in ks_list:
                rec, ndcg = batch_performance(pred.detach().cpu(), batch["label"].detach().cpu(), k)
                col = ks_list.index(k)
                train_recall[i, col] = rec
                train_ndcg[i, col] = ndcg

        logging.info("Train loss: %.4f", train_loss / max(1, len(train_loader)))
        for k in ks_list:
            col = ks_list.index(k)
            logging.info("Train Recall@%d: %.4f", k, float(train_recall[:, col].mean()))
            logging.info("Train NDCG@%d: %.4f", k, float(train_ndcg[:, col].mean()))

        # -------- 测试阶段 --------
        model.eval()
        test_loss = 0.0
        test_recall = np.zeros((len(test_loader), len(ks_list)), dtype=np.float32)
        test_ndcg = np.zeros((len(test_loader), len(ks_list)), dtype=np.float32)
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                pred, _ = model(test_dataset, batch)
                loss_next = criterion(pred, batch["label"])
                test_loss += loss_next.item()

                for k in ks_list:
                    rec, ndcg = batch_performance(pred.detach().cpu(), batch["label"].detach().cpu(), k)
                    col = ks_list.index(k)
                    test_recall[i, col] = rec
                    test_ndcg[i, col] = ndcg

        logging.info("Test loss: %.4f", test_loss / max(1, len(test_loader)))
        for k in ks_list:
            col = ks_list.index(k)
            logging.info("Test Recall@%d: %.4f", k, float(test_recall[:, col].mean()))
            logging.info("Test NDCG@%d: %.4f", k, float(test_ndcg[:, col].mean()))

        # 按 Recall@5 选择最优 checkpoint
        rec5 = float(test_recall[:, 1].mean())
        if rec5 > best_rec5:
            best_rec5 = rec5
            saved_model_path = os.path.join(current_save_dir, "{}_T.pt".format(args.dataset))
            torch.save(model.state_dict(), saved_model_path)
            logging.info("Saved best model at epoch %d", epoch)

        logging.info("Epoch time: %.2f min", (time.time() - start_time) / 60.0)

    logging.info("Best Test Recall@5: %.4f", best_rec5)


if __name__ == "__main__":
    main()
