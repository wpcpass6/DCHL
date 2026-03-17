# coding=utf-8
"""
THGR-Next 训练入口。

选模指标：NDCG@10（相较仅用 Recall@5 更稳）。
"""

import argparse
import csv
import datetime
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from metrics import batch_performance
from TH_dataset_next import THNextDataset, th_collate_fn
from TH_hypergraph_builder import build_and_save
from TH_model_next import THNextModel


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="TKY", choices=["NYC", "TKY"])
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--deviceID", type=int, default=0)

    p.add_argument("--data_dir", type=str, default="datasets")
    p.add_argument("--poi_cat_filename", type=str, default="")
    p.add_argument("--region_grid_size", type=float, default=0.02)
    p.add_argument("--geo_k", type=int, default=15)
    p.add_argument("--knn_chunk_size", type=int, default=512)

    p.add_argument("--use_func", action="store_true")
    p.add_argument("--use_region", action="store_true")
    p.add_argument("--use_geo", action="store_true")
    p.add_argument("--use_mob", action="store_true")
    p.add_argument("--mob_window", type=int, default=5)
    p.add_argument("--mob_min_freq", type=int, default=2)

    p.add_argument("--num_hg_layers", type=int, default=1)
    p.add_argument("--n_tf_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=256)

    p.add_argument("--alpha_cat", type=float, default=0.2)
    p.add_argument("--beta_region", type=float, default=0.2)
    p.add_argument("--mask_ratio", type=float, default=0.05)
    p.add_argument("--lambda_mask", type=float, default=0.03)
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--min_delta", type=float, default=1e-4)

    p.add_argument("--save_dir", type=str, default="logs")
    args = p.parse_args()

    # 默认开启 func/region/geo，mobility 默认关闭
    if (not args.use_func) and (not args.use_region) and (not args.use_geo) and (not args.use_mob):
        args.use_func = True
        args.use_region = True
        args.use_geo = True
        args.use_mob = False
    return args


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.deviceID}" if torch.cuda.is_available() else "cpu")

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = os.path.join(args.save_dir, "TH_" + current_time)
    os.makedirs(save_dir, exist_ok=True)

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "log_training.txt"),
        filemode="w+",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger("").addHandler(console)

    with open(os.path.join(save_dir, f"{args.dataset}_TH_args.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    if args.dataset == "TKY":
        num_users, num_pois = 2173, 7038
    else:
        num_users, num_pois = 834, 3835
    padding_idx = num_pois

    logging.info("1. Build/Load Hypergraph")
    graph_path, graph_obj = build_and_save(args.dataset, args)
    logging.info("hypergraph path: %s", graph_path)
    logging.info("nodes=%d edges=%d", graph_obj["num_nodes"], graph_obj["num_edges"])

    logging.info("2. Load Dataset")
    train_dataset = THNextDataset(
        data_filename=os.path.join(args.data_dir, args.dataset, "train_poi_zero.txt"),
        num_users=num_users,
        num_pois=num_pois,
        padding_idx=padding_idx,
        args=args,
        device=device,
        graph_obj=graph_obj,
    )
    test_dataset = THNextDataset(
        data_filename=os.path.join(args.data_dir, args.dataset, "test_poi_zero.txt"),
        num_users=num_users,
        num_pois=num_pois,
        padding_idx=padding_idx,
        args=args,
        device=device,
        graph_obj=graph_obj,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: th_collate_fn(b, padding_value=padding_idx, max_seq_len=args.max_seq_len),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: th_collate_fn(b, padding_value=padding_idx, max_seq_len=args.max_seq_len),
    )

    logging.info("3. Init Model")
    model = THNextModel(
        num_users=num_users,
        num_pois=num_pois,
        num_cats=train_dataset.num_cats,
        num_regions=train_dataset.num_regions,
        args=args,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    ks = [1, 5, 10, 20]

    best_ndcg10 = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    best_metrics = {
        "test_recall1": 0.0,
        "test_recall5": 0.0,
        "test_recall10": 0.0,
        "test_recall20": 0.0,
        "test_ndcg1": 0.0,
        "test_ndcg5": 0.0,
        "test_ndcg10": 0.0,
        "test_ndcg20": 0.0,
    }

    logging.info("4. Start Training")
    for epoch in range(args.num_epochs):
        st = time.time()
        model.train()
        train_loss = 0.0
        train_rec = np.zeros((len(train_loader), len(ks)), dtype=np.float32)
        train_nd = np.zeros((len(train_loader), len(ks)), dtype=np.float32)

        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pred, loss_mask = model(train_dataset, batch)
            loss_next = criterion(pred, batch["label"])
            loss = loss_next + args.lambda_mask * loss_mask
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            for k in ks:
                rec, nd = batch_performance(pred.detach().cpu(), batch["label"].detach().cpu(), k)
                c = ks.index(k)
                train_rec[i, c] = rec
                train_nd[i, c] = nd

        model.eval()
        test_loss = 0.0
        test_rec = np.zeros((len(test_loader), len(ks)), dtype=np.float32)
        test_nd = np.zeros((len(test_loader), len(ks)), dtype=np.float32)
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                pred, _ = model(test_dataset, batch)
                loss_next = criterion(pred, batch["label"])
                test_loss += loss_next.item()
                for k in ks:
                    rec, nd = batch_performance(pred.detach().cpu(), batch["label"].detach().cpu(), k)
                    c = ks.index(k)
                    test_rec[i, c] = rec
                    test_nd[i, c] = nd

        logging.info("================= Epoch %d/%d =================", epoch, args.num_epochs)
        logging.info("Train loss: %.4f", train_loss / max(1, len(train_loader)))
        for k in ks:
            c = ks.index(k)
            logging.info("Train Recall@%d: %.4f", k, float(train_rec[:, c].mean()))
            logging.info("Train NDCG@%d: %.4f", k, float(train_nd[:, c].mean()))

        logging.info("Test loss: %.4f", test_loss / max(1, len(test_loader)))
        for k in ks:
            c = ks.index(k)
            logging.info("Test Recall@%d: %.4f", k, float(test_rec[:, c].mean()))
            logging.info("Test NDCG@%d: %.4f", k, float(test_nd[:, c].mean()))

        ndcg10 = float(test_nd[:, ks.index(10)].mean())
        rec10 = float(test_rec[:, ks.index(10)].mean())
        scheduler.step(ndcg10)
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info("LR: %.6e", current_lr)
        logging.info(
            "EPOCH_KEY epoch=%d ndcg10=%.4f recall10=%.4f best_ndcg10=%.4f",
            epoch,
            ndcg10,
            rec10,
            best_ndcg10,
        )

        if ndcg10 > best_ndcg10:
            best_ndcg10 = ndcg10
            best_epoch = epoch
            no_improve_epochs = 0
            best_metrics = {
                "test_recall1": float(test_rec[:, ks.index(1)].mean()),
                "test_recall5": float(test_rec[:, ks.index(5)].mean()),
                "test_recall10": rec10,
                "test_recall20": float(test_rec[:, ks.index(20)].mean()),
                "test_ndcg1": float(test_nd[:, ks.index(1)].mean()),
                "test_ndcg5": float(test_nd[:, ks.index(5)].mean()),
                "test_ndcg10": ndcg10,
                "test_ndcg20": float(test_nd[:, ks.index(20)].mean()),
            }
            model_path = os.path.join(save_dir, f"{args.dataset}_TH_next.pt")
            torch.save(model.state_dict(), model_path)
            logging.info("Saved best model at epoch %d by NDCG@10", epoch)
        else:
            improve = ndcg10 - best_ndcg10
            if improve < args.min_delta:
                no_improve_epochs += 1

        if no_improve_epochs >= args.early_stop_patience:
            logging.info(
                "Early stop at epoch %d (patience=%d, min_delta=%.1e)",
                epoch,
                args.early_stop_patience,
                args.min_delta,
            )
            break

        logging.info("Epoch time: %.2f min", (time.time() - st) / 60.0)

    logging.info("Best epoch: %d", best_epoch)
    logging.info("Best Test NDCG@10: %.4f", best_ndcg10)

    # 让网格实验结果一眼可读：终端/日志输出单行 KEY，并写入 run 内 yaml + 全局 csv。
    key_line = (
        "BEST_RESULT "
        f"run_dir={save_dir} dataset={args.dataset} "
        f"lambda_mask={args.lambda_mask} geo_k={args.geo_k} num_hg_layers={args.num_hg_layers} "
        f"alpha_cat={args.alpha_cat} beta_region={args.beta_region} "
        f"best_epoch={best_epoch} "
        f"best_ndcg10={best_metrics['test_ndcg10']:.4f} best_recall10={best_metrics['test_recall10']:.4f}"
    )
    logging.info("=" * 80)
    logging.info(key_line)
    logging.info("=" * 80)

    best_obj = {
        "run_dir": save_dir,
        "dataset": args.dataset,
        "lambda_mask": float(args.lambda_mask),
        "geo_k": int(args.geo_k),
        "num_hg_layers": int(args.num_hg_layers),
        "alpha_cat": float(args.alpha_cat),
        "beta_region": float(args.beta_region),
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
    }
    with open(os.path.join(save_dir, "best_result.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(best_obj, f, sort_keys=False)

    csv_path = os.path.join(args.save_dir, "TH_grid_results.csv")
    csv_header = [
        "run_dir",
        "dataset",
        "lambda_mask",
        "geo_k",
        "num_hg_layers",
        "alpha_cat",
        "beta_region",
        "best_epoch",
        "best_ndcg10",
        "best_recall10",
        "best_ndcg5",
        "best_recall5",
        "best_ndcg20",
        "best_recall20",
    ]
    csv_row = {
        "run_dir": save_dir,
        "dataset": args.dataset,
        "lambda_mask": float(args.lambda_mask),
        "geo_k": int(args.geo_k),
        "num_hg_layers": int(args.num_hg_layers),
        "alpha_cat": float(args.alpha_cat),
        "beta_region": float(args.beta_region),
        "best_epoch": int(best_epoch),
        "best_ndcg10": round(best_metrics["test_ndcg10"], 6),
        "best_recall10": round(best_metrics["test_recall10"], 6),
        "best_ndcg5": round(best_metrics["test_ndcg5"], 6),
        "best_recall5": round(best_metrics["test_recall5"], 6),
        "best_ndcg20": round(best_metrics["test_ndcg20"], 6),
        "best_recall20": round(best_metrics["test_recall20"], 6),
    }
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        if need_header:
            writer.writeheader()
        writer.writerow(csv_row)


if __name__ == "__main__":
    main()
