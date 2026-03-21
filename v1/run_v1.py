# coding=utf-8
"""
V1 训练入口：
- Time-Category 视图 + POI Mask Fill
- 保留 DCHL 原有推荐与对比学习主干
"""

import argparse
import datetime
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from metrics import batch_performance
from v1.dataset_v1 import POIDatasetV1, POIPartialDatasetV1, collate_fn_4sq, infer_user_poi_num
from v1.model_v1 import DCHL_V1


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def setup_logger(save_dir):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=str(Path(save_dir) / 'log_training.txt'),
        filemode='w+'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TKY', help='数据集名，仅用于文件名拼接')
    parser.add_argument('--data_dir', default='datasets/TKY_v1', help='预处理后数据目录')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--distance_threshold', default=2.5, type=float)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='从训练用户中划出验证集比例，用于早停与选模型')
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--deviceID', type=int, default=0)

    parser.add_argument('--lambda_cl', type=float, default=0.1)
    parser.add_argument('--lambda_mask', type=float, default=0.05)

    parser.add_argument('--num_mv_layers', type=int, default=3)
    parser.add_argument('--num_geo_layers', type=int, default=3)
    parser.add_argument('--num_di_layers', type=int, default=3)
    parser.add_argument('--num_tc_layers', type=int, default=2)

    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--keep_rate', type=float, default=1.0)
    parser.add_argument('--keep_rate_poi', type=float, default=1.0)
    parser.add_argument('--keep_rate_tc', type=float, default=1.0)
    parser.add_argument('--tc_weight_mode', type=str, default='log1p', choices=['log1p', 'binary'])
    parser.add_argument('--beta_tc', type=float, default=1.0, help='TC 视图在 POI 融合中的权重')

    parser.add_argument('--mask_rate', type=float, default=0.2)
    parser.add_argument('--mask_alpha', type=int, default=2)

    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='logs_v1')

    # 早停相关参数
    parser.add_argument('--early_stop_patience', type=int, default=3,
                        help='验证指标连续多少个epoch不提升后停止训练')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                        help='判定“有提升”的最小阈值')
    parser.add_argument('--early_stop_metric', type=str, default='NDCG10',
                        choices=['NDCG10', 'Rec10', 'NDCG5', 'Rec5'],
                        help='用于早停和保存best模型的验证指标')
    return parser


def split_train_val_user_indices(num_users, val_ratio, seed):
    """
    把训练用户切分为 train/val 索引。

    说明：
    - 这里按用户随机切分（固定 seed 可复现）；
    - 保证 val 至少 1 个用户，train 至少 1 个用户。
    """
    indices = np.arange(num_users)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_val = int(num_users * val_ratio)
    n_val = max(1, n_val)
    n_val = min(num_users - 1, n_val)

    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()
    return train_indices, val_indices


def evaluate_epoch(model, graph_dataset, dataloader, criterion, device, ks_list, args):
    """
    评估一个 dataloader，返回平均loss与各指标。

    参数说明：
    - graph_dataset: 提供图结构（HG/geo graph 等）的完整数据集对象；
                     train/val 都应复用“训练图”，test 用测试图。
    """
    model.eval()
    total_loss = 0.0
    recall_array = np.zeros((len(dataloader), len(ks_list)))
    ndcg_array = np.zeros((len(dataloader), len(ks_list)))

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            predictions, loss_cl_users, loss_cl_pois, loss_mask = model(graph_dataset, batch)
            loss_rec = criterion(predictions, batch["label"].to(device))
            loss = loss_rec + args.lambda_cl * (loss_cl_users + loss_cl_pois) + args.lambda_mask * loss_mask
            total_loss += loss.item()

            for k in ks_list:
                recall, ndcg = batch_performance(predictions.detach().cpu(), batch["label"].detach().cpu(), k)
                cidx = ks_list.index(k)
                recall_array[idx, cidx] = recall
                ndcg_array[idx, cidx] = ndcg

    results = {}
    for k in ks_list:
        cidx = ks_list.index(k)
        results[f"Rec{k}"] = float(np.mean(recall_array[:, cidx]))
        results[f"NDCG{k}"] = float(np.mean(ndcg_array[:, cidx]))

    avg_loss = total_loss / len(dataloader)
    return avg_loss, results


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 设备
    device = torch.device(f"cuda:{args.deviceID}" if torch.cuda.is_available() else "cpu")

    # 输出目录
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    current_save_dir = save_root / f"{args.dataset}_V1_{current_time}"
    current_save_dir.mkdir(parents=True, exist_ok=False)

    setup_logger(current_save_dir)
    with open(current_save_dir / f"{args.dataset}_v1_args.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(vars(args), f, sort_keys=False, allow_unicode=True)

    data_dir = Path(args.data_dir)
    num_users, num_pois = infer_user_poi_num(data_dir, args.dataset)
    padding_idx = num_pois

    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info(f"device: {device}")
    logging.info(f"num_users={num_users}, num_pois={num_pois}")

    # 路径
    train_file = str(data_dir / "train_poi_zero.txt")
    test_file = str(data_dir / "test_poi_zero.txt")
    coos_file = str(data_dir / f"{args.dataset}_pois_coos_poi_zero.pkl")
    cat_file = str(data_dir / f"{args.dataset}_poi_cat.pkl")
    train_slot_file = str(data_dir / "train_time_slot.pkl")
    test_slot_file = str(data_dir / "test_time_slot.pkl")

    logging.info("2. Load Dataset")
    # 训练图：用于 train/val（避免在验证阶段看到测试图结构）
    train_dataset = POIDatasetV1(
        data_filename=train_file,
        pois_coos_filename=coos_file,
        poi_cat_filename=cat_file,
        time_slot_filename=train_slot_file,
        num_users=num_users,
        num_pois=num_pois,
        padding_idx=padding_idx,
        args=args,
        device=device,
    )
    # 测试图：用于最终 test 评估
    test_dataset = POIDatasetV1(
        data_filename=test_file,
        pois_coos_filename=coos_file,
        poi_cat_filename=cat_file,
        time_slot_filename=test_slot_file,
        num_users=num_users,
        num_pois=num_pois,
        padding_idx=padding_idx,
        args=args,
        device=device,
    )

    logging.info("3. Construct DataLoader")
    train_indices, val_indices = split_train_val_user_indices(
        num_users=num_users,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_partial = POIPartialDatasetV1(train_dataset, train_indices)
    val_partial = POIPartialDatasetV1(train_dataset, val_indices)

    logging.info(f"Train users: {len(train_partial)} | Val users: {len(val_partial)} | Test users: {len(test_dataset)}")

    train_loader = DataLoader(
        dataset=train_partial,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=padding_idx)
    )
    val_loader = DataLoader(
        dataset=val_partial,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=padding_idx)
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=padding_idx)
    )

    logging.info("4. Load Model")
    model = DCHL_V1(num_users, num_pois, args, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_scheduler_factor)

    logging.info("5. Start Training")
    ks_list = [1, 5, 10, 20]
    final_results = {
        "Rec1": 0.0, "Rec5": 0.0, "Rec10": 0.0, "Rec20": 0.0,
        "NDCG1": 0.0, "NDCG5": 0.0, "NDCG10": 0.0, "NDCG20": 0.0,
    }

    # 早停状态
    best_val_metric = -float('inf')
    no_improve_epochs = 0
    monitor_loss = float('inf')
    best_model_path = current_save_dir / f"{args.dataset}_V1_best.pt"

    def pick_metric(result_dict):
        return result_dict[args.early_stop_metric]

    for epoch in range(args.num_epochs):
        logging.info(f"================= Epoch {epoch}/{args.num_epochs} =================")
        start_time = time.time()

        # ---------- 训练 ----------
        model.train()
        train_loss = 0.0
        train_recall_array = np.zeros((len(train_loader), len(ks_list)))
        train_ndcg_array = np.zeros((len(train_loader), len(ks_list)))

        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            predictions, loss_cl_users, loss_cl_pois, loss_mask = model(train_dataset, batch)

            loss_rec = criterion(predictions, batch["label"].to(device))
            loss = loss_rec + args.lambda_cl * (loss_cl_users + loss_cl_pois) + args.lambda_mask * loss_mask

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            for k in ks_list:
                recall, ndcg = batch_performance(predictions.detach().cpu(), batch["label"].detach().cpu(), k)
                cidx = ks_list.index(k)
                train_recall_array[idx, cidx] = recall
                train_ndcg_array[idx, cidx] = ndcg

            logging.info(
                f"Train Batch {idx}/{len(train_loader)} "
                f"loss_rec={loss_rec.item():.4f} "
                f"loss_cl_u={loss_cl_users.item():.4f} "
                f"loss_cl_p={loss_cl_pois.item():.4f} "
                f"loss_mask={loss_mask.item():.4f} "
                f"loss={loss.item():.4f}"
            )

        logging.info(f"Training epoch time: {(time.time() - start_time) / 60:.2f} min")
        logging.info(f"Training loss: {train_loss / len(train_loader):.4f}")
        for k in ks_list:
            cidx = ks_list.index(k)
            logging.info(f"Train Recall@{k}: {np.mean(train_recall_array[:, cidx]):.4f}")
            logging.info(f"Train NDCG@{k}: {np.mean(train_ndcg_array[:, cidx]):.4f}")

        # ---------- 验证 ----------
        val_loss, val_results = evaluate_epoch(
            model=model,
            graph_dataset=train_dataset,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            ks_list=ks_list,
            args=args,
        )
        logging.info(f"Val loss: {val_loss:.4f}")
        for k in ks_list:
            logging.info(f"Val Recall@{k}: {val_results[f'Rec{k}']:.4f}")
            logging.info(f"Val NDCG@{k}: {val_results[f'NDCG{k}']:.4f}")

        # ---------- 测试（仅用于观察，不参与早停） ----------
        test_loss, test_results = evaluate_epoch(
            model=model,
            graph_dataset=test_dataset,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            ks_list=ks_list,
            args=args,
        )
        logging.info(f"Test loss: {test_loss:.4f}")
        for k in ks_list:
            logging.info(f"Test Recall@{k}: {test_results[f'Rec{k}']:.4f}")
            logging.info(f"Test NDCG@{k}: {test_results[f'NDCG{k}']:.4f}")

        monitor_loss = min(monitor_loss, val_loss)
        scheduler.step(monitor_loss)

        # ---------- 基于验证集早停与保存best ----------
        current_val_metric = pick_metric(val_results)
        if current_val_metric > best_val_metric + args.early_stop_min_delta:
            best_val_metric = current_val_metric
            no_improve_epochs = 0
            torch.save(model.state_dict(), str(best_model_path))
            logging.info(
                f"Update best model at epoch {epoch} | "
                f"best {args.early_stop_metric}={best_val_metric:.4f}"
            )
        else:
            no_improve_epochs += 1
            logging.info(
                f"No improvement on {args.early_stop_metric} for {no_improve_epochs} epoch(s)."
            )

        # 记录过程中的 test 最好值（仅便于观察）
        for metric_key in final_results:
            final_results[metric_key] = max(final_results[metric_key], test_results[metric_key])

        logging.info("==================================\n")

        if no_improve_epochs >= args.early_stop_patience:
            logging.info(
                f"Early stopped at epoch {epoch}. "
                f"Best val {args.early_stop_metric}={best_val_metric:.4f}"
            )
            break

    # 训练结束后，用 best-val 模型做一次最终 test
    if best_model_path.exists():
        model.load_state_dict(torch.load(str(best_model_path), map_location=device))
        best_test_loss, best_test_results = evaluate_epoch(
            model=model,
            graph_dataset=test_dataset,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            ks_list=ks_list,
            args=args,
        )
        logging.info("7. Final Test Results (Best-VAL Model)")
        logging.info(f"Final Test loss: {best_test_loss:.4f}")
        logging.info({k: f"{v:.4f}" for k, v in best_test_results.items()})
    else:
        logging.info("7. Final Test Results skipped: best model not found")

    logging.info("6. Final Results")
    logging.info({k: f"{v:.4f}" for k, v in final_results.items()})


if __name__ == '__main__':
    main()
