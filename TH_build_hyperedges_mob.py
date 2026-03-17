# coding=utf-8
"""
构建移动视角超边（仅使用 train 数据，避免未来信息泄漏）。
"""

import argparse
import os
from collections import Counter

from TH_utils import load_pkl, save_pkl


def build_mobility_hyperedges(train_pkl, window_size=5, min_freq=2):
    data = load_pkl(train_pkl)
    sessions_dict = data[0]
    mob_counter = Counter()

    for _uid, sessions in sessions_dict.items():
        traj = []
        for s in sessions:
            traj.extend(s)
        if len(traj) < window_size:
            continue
        for i in range(len(traj) - window_size + 1):
            win = tuple(sorted(set(traj[i : i + window_size])))
            if len(win) >= 2:
                mob_counter[win] += 1

    if len(mob_counter) == 0:
        return []

    max_freq = max(mob_counter.values())
    edges = []
    for nodes, freq in mob_counter.items():
        if freq < min_freq:
            continue
        w = float(freq / max_freq)
        edges.append({"edge_type": "mob", "nodes": list(nodes), "weight": w, "freq": int(freq)})
    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["NYC", "TKY"], required=True)
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--min_freq", type=int, default=2)
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_pkl = os.path.join(dataset_dir, "train_poi_zero.txt")

    edges = build_mobility_hyperedges(train_pkl, window_size=args.window_size, min_freq=args.min_freq)
    out = {
        "window_size": args.window_size,
        "min_freq": args.min_freq,
        "mob_edges": edges,
    }
    out_path = os.path.join(dataset_dir, f"TH_hyperedges_mob_{args.dataset}.pkl")
    save_pkl(out_path, out)
    print("[TH_build_hyperedges_mob] saved:", out_path, "num_edges=", len(edges))


if __name__ == "__main__":
    main()
