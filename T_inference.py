# coding=utf-8
"""
THGR 推理脚本（对应 T_run.py 的训练输出）。

用途：
1) 加载指定训练目录中的参数与模型权重；
2) 在测试集上计算 Recall@K / NDCG@K；
3) 可选仅评估 active user 子集。
"""

import argparse
import logging
import os
import random
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import batch_performance
from T_dataset import TPOIDataset, t_collate_fn
from T_model import THGR
from T_utils import load_dict_from_pkl


# 清理并启用 cudnn 加速
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="TKY", help="NYC/TKY")
parser.add_argument("--deviceID", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="logs")
parser.add_argument("--saved_model_path", type=str, required=True, help="训练目录名，例如 T_20260317_101010")
parser.add_argument("--eval_active_only", action="store_true", help="仅评估 active_user_dict 中用户")

# 下面这些参数用于构图/模型结构，必须与训练时一致。
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--region_grid_size", type=float, default=0.02)
parser.add_argument("--cat_grid_size", type=float, default=0.01)
parser.add_argument("--geo_k", type=int, default=20)
parser.add_argument("--knn_chunk_size", type=int, default=512)
parser.add_argument("--num_hg_layers", type=int, default=2)
parser.add_argument("--poi_cat_filename", type=str, default="", help="可选：POI类别pkl文件（dict或list）")
parser.add_argument("--n_tf_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--max_seq_len", type=int, default=256)
parser.add_argument("--alpha_cat", type=float, default=0.2)
parser.add_argument("--beta_region", type=float, default=0.2)
parser.add_argument("--mask_ratio", type=float, default=0.05)
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:{}".format(args.deviceID) if torch.cuda.is_available() else "cpu")


def _load_train_args_if_exists(run_dir):
    """若训练目录中存在 yaml 参数文件，则回填关键结构参数。

    这样可避免“推理参数与训练参数不一致”导致的 shape mismatch。
    """
    yaml_path = os.path.join(run_dir, f"{args.dataset}_T_args.yaml")
    if not os.path.exists(yaml_path):
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    for key in [
        "emb_dim",
        "hidden_dim",
        "dropout",
        "region_grid_size",
        "cat_grid_size",
        "geo_k",
        "knn_chunk_size",
        "num_hg_layers",
        "poi_cat_filename",
        "n_tf_layers",
        "n_heads",
        "max_seq_len",
        "alpha_cat",
        "beta_region",
        "mask_ratio",
    ]:
        if key in train_cfg:
            setattr(args, key, train_cfg[key])


def main():
    # ---------- 日志初始化 ----------
    run_dir = os.path.join(args.save_dir, args.saved_model_path)
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"saved_model_path not found: {run_dir}")

    _load_train_args_if_exists(run_dir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(run_dir, "log_inference.txt"),
        filemode="w+",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info("device: %s", device)

    # ---------- 数据集规格 ----------
    if args.dataset == "TKY":
        num_users = 2173
        num_pois = 7038
        padding_idx = num_pois
    elif args.dataset == "NYC":
        num_users = 834
        num_pois = 3835
        padding_idx = num_pois
    else:
        raise ValueError("Only NYC/TKY are supported in T_inference.py")

    # ---------- 加载测试集 ----------
    logging.info("2. Load Test Dataset")
    test_dataset = TPOIDataset(
        data_filename="datasets/{}/test_poi_zero.txt".format(args.dataset),
        pois_coos_filename="datasets/{}/{}_pois_coos_poi_zero.pkl".format(args.dataset, args.dataset),
        num_users=num_users,
        num_pois=num_pois,
        padding_idx=padding_idx,
        args=args,
        device=device,
    )

    # 可选：仅 active 用户评估
    if args.eval_active_only:
        active_path = "datasets/{}/active_user_dict.pkl".format(args.dataset)
        if not os.path.exists(active_path):
            raise FileNotFoundError(f"active user file not found: {active_path}")
        active_user_dict = load_dict_from_pkl(active_path)
        active_user_indices = set(active_user_dict.keys())
        all_indices = [i for i in range(len(test_dataset)) if i in active_user_indices]
    else:
        all_indices = list(range(len(test_dataset)))

    # 子集包装（沿用原始 Dataset 输出）
    subset_samples = [test_dataset[i] for i in all_indices]

    logging.info("3. Construct DataLoader")
    test_loader = DataLoader(
        dataset=subset_samples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: t_collate_fn(b, padding_value=padding_idx, max_seq_len=args.max_seq_len),
    )

    # ---------- 初始化模型并加载权重 ----------
    logging.info("4. Load Model")
    model = THGR(
        num_users=num_users,
        num_pois=num_pois,
        num_cats=test_dataset.num_cats,
        num_regions=test_dataset.num_regions,
        args=args,
        device=device,
    ).to(device)

    ckpt_path = os.path.join(run_dir, "{}_T.pt".format(args.dataset))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # ---------- 推理评估 ----------
    logging.info("5. Start Inference")
    model.eval()
    ks_list = [1, 5, 10, 20]
    test_recall = np.zeros((len(test_loader), len(ks_list)), dtype=np.float32)
    test_ndcg = np.zeros((len(test_loader), len(ks_list)), dtype=np.float32)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            pred, _ = model(test_dataset, batch)

            for k in ks_list:
                rec, ndcg = batch_performance(pred.detach().cpu(), batch["label"].detach().cpu(), k)
                col = ks_list.index(k)
                test_recall[i, col] = rec
                test_ndcg[i, col] = ndcg

    logging.info("Testing results:")
    for k in ks_list:
        col = ks_list.index(k)
        logging.info("Recall@%d: %.4f", k, float(test_recall[:, col].mean()))
        logging.info("NDCG@%d: %.4f", k, float(test_ndcg[:, col].mean()))


if __name__ == "__main__":
    main()
