# coding=utf-8
"""
构建功能视角超边（category hyperedges）。
"""

import argparse
import os
from collections import defaultdict

import numpy as np

from TH_utils import load_pkl, save_pkl


def build_function_hyperedges(num_pois, poi_cat_pkl):
    """返回功能超边列表。

    每条超边结构：
    {
      "edge_type": "func",
      "nodes": [poi_local_ids],
      "weight": float
    }
    """
    cat_obj = load_pkl(poi_cat_pkl)
    if isinstance(cat_obj, dict):
        raw = [cat_obj[i] for i in range(num_pois)]
    else:
        raw = list(cat_obj)[:num_pois]

    uniq = {c: i for i, c in enumerate(sorted(set(raw)))}
    poi_to_cat = np.array([uniq[c] for c in raw], dtype=np.int32)
    num_cats = len(uniq)

    groups = defaultdict(list)
    for poi in range(num_pois):
        groups[int(poi_to_cat[poi])].append(poi)

    max_size = max(len(v) for v in groups.values()) if groups else 1
    edges = []
    for c in range(num_cats):
        pois = groups[c]
        # 按组内规模归一化（避免超大类边权过强）
        w = float(len(pois) / max_size)
        edges.append({"edge_type": "func", "cat_id": c, "nodes": pois, "weight": w})
    return edges, poi_to_cat, num_cats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["NYC", "TKY"], required=True)
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--poi_cat_filename", type=str, default="")
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    poi_coord_pkl = os.path.join(dataset_dir, f"{args.dataset}_pois_coos_poi_zero.pkl")
    poi_coords = load_pkl(poi_coord_pkl)
    num_pois = len(poi_coords)

    poi_cat_file = args.poi_cat_filename or f"{args.dataset}_poi_cat.pkl"
    if not os.path.isabs(poi_cat_file):
        poi_cat_file = os.path.join(dataset_dir, poi_cat_file)

    edges, poi_to_cat, num_cats = build_function_hyperedges(num_pois, poi_cat_file)
    out = {
        "num_pois": num_pois,
        "num_cats": num_cats,
        "poi_to_cat": poi_to_cat,
        "func_edges": edges,
    }
    out_path = os.path.join(dataset_dir, f"TH_hyperedges_func_{args.dataset}.pkl")
    save_pkl(out_path, out)
    print("[TH_build_hyperedges_func] saved:", out_path)


if __name__ == "__main__":
    main()
