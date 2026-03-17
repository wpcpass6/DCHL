# coding=utf-8
"""
THGR-Next 推理脚本。
"""

import argparse
import logging
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from metrics import batch_performance
from TH_dataset_next import THNextDataset, th_collate_fn
from TH_hypergraph_builder import build_and_save
from TH_model_next import THNextModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="TKY", choices=["NYC", "TKY"])
    p.add_argument("--deviceID", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="logs")
    p.add_argument("--saved_model_path", type=str, required=True)

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

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num_hg_layers", type=int, default=1)
    p.add_argument("--n_tf_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--alpha_cat", type=float, default=0.2)
    p.add_argument("--beta_region", type=float, default=0.2)
    p.add_argument("--mask_ratio", type=float, default=0.05)
    args = p.parse_args()

    if (not args.use_func) and (not args.use_region) and (not args.use_geo) and (not args.use_mob):
        args.use_func = True
        args.use_region = True
        args.use_geo = True
        args.use_mob = False
    return args


def _load_train_args(run_dir, args):
    yaml_path = os.path.join(run_dir, f"{args.dataset}_TH_args.yaml")
    if not os.path.exists(yaml_path):
        return args
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.deviceID}" if torch.cuda.is_available() else "cpu")

    run_dir = os.path.join(args.save_dir, args.saved_model_path)
    if not os.path.exists(run_dir):
        raise FileNotFoundError(run_dir)
    args = _load_train_args(run_dir, args)

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(run_dir, "log_inference_next.txt"),
        filemode="w+",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    if args.dataset == "TKY":
        num_users, num_pois = 2173, 7038
    else:
        num_users, num_pois = 834, 3835
    padding_idx = num_pois

    graph_path, graph_obj = build_and_save(args.dataset, args)
    logging.info("graph path: %s", graph_path)

    test_dataset = THNextDataset(
        data_filename=os.path.join(args.data_dir, args.dataset, "test_poi_zero.txt"),
        num_users=num_users,
        num_pois=num_pois,
        padding_idx=padding_idx,
        args=args,
        device=device,
        graph_obj=graph_obj,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: th_collate_fn(b, padding_value=padding_idx, max_seq_len=args.max_seq_len),
    )

    model = THNextModel(
        num_users=num_users,
        num_pois=num_pois,
        num_cats=test_dataset.num_cats,
        num_regions=test_dataset.num_regions,
        args=args,
        device=device,
    ).to(device)

    model_path = os.path.join(run_dir, f"{args.dataset}_TH_next.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    ks = [1, 5, 10, 20]
    rec = np.zeros((len(test_loader), len(ks)), dtype=np.float32)
    nd = np.zeros((len(test_loader), len(ks)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            pred, _ = model(test_dataset, batch)
            for k in ks:
                r, n = batch_performance(pred.detach().cpu(), batch["label"].detach().cpu(), k)
                c = ks.index(k)
                rec[i, c] = r
                nd[i, c] = n

    for k in ks:
        c = ks.index(k)
        logging.info("Recall@%d: %.4f", k, float(rec[:, c].mean()))
        logging.info("NDCG@%d: %.4f", k, float(nd[:, c].mean()))


if __name__ == "__main__":
    main()
