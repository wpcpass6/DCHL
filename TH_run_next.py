# coding=utf-8
"""
THGR-Next 训练入口。

选模指标：NDCG@10（相较仅用 Recall@5 更稳）。
"""

import argparse
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
    criterion = nn.CrossEntropyLoss().to(device)
    ks = [1, 5, 10, 20]

    best_ndcg10 = -1.0
    best_epoch = -1

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
        if ndcg10 > best_ndcg10:
            best_ndcg10 = ndcg10
            best_epoch = epoch
            model_path = os.path.join(save_dir, f"{args.dataset}_TH_next.pt")
            torch.save(model.state_dict(), model_path)
            logging.info("Saved best model at epoch %d by NDCG@10", epoch)

        logging.info("Epoch time: %.2f min", (time.time() - st) / 60.0)

    logging.info("Best epoch: %d", best_epoch)
    logging.info("Best Test NDCG@10: %.4f", best_ndcg10)


if __name__ == "__main__":
    main()
