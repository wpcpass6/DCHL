# coding=utf-8
"""
构建地理视角超边：
1) region hyperedges
2) local spatial hyperedges
"""

import argparse
import os
from collections import defaultdict

import numpy as np

from TH_utils import load_poi_coords, save_pkl, build_region_ids, build_knn_neighbors


def build_geo_hyperedges(poi_coords, region_grid_size=0.02, geo_k=15, knn_chunk_size=512):
    num_pois = len(poi_coords)
    poi_to_region, num_regions = build_region_ids(poi_coords, region_grid_size=region_grid_size)

    groups = defaultdict(list)
    for poi in range(num_pois):
        groups[int(poi_to_region[poi])].append(poi)

    max_size = max(len(v) for v in groups.values()) if groups else 1
    region_edges = []
    for r in range(num_regions):
        pois = groups[r]
        w = float(len(pois) / max_size)
        region_edges.append({"edge_type": "region", "region_id": r, "nodes": pois, "weight": w})

    neighbors, neighbor_dists = build_knn_neighbors(poi_coords, k=geo_k, chunk_size=knn_chunk_size)
    geo_edges = []
    for poi in range(num_pois):
        nbs = [int(x) for x in neighbors[poi].tolist()]
        # 距离越近权重越大，取 1/(1+avg_dist)
        avg_dist = float(np.mean(neighbor_dists[poi]))
        w = 1.0 / (1.0 + avg_dist)
        geo_edges.append(
            {
                "edge_type": "geo",
                "poi_id": poi,
                "region_id": int(poi_to_region[poi]),
                "neighbors": nbs,
                "weight": w,
            }
        )

    return region_edges, geo_edges, poi_to_region, num_regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["NYC", "TKY"], required=True)
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--region_grid_size", type=float, default=0.02)
    parser.add_argument("--geo_k", type=int, default=15)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    poi_coord_pkl = os.path.join(dataset_dir, f"{args.dataset}_pois_coos_poi_zero.pkl")
    poi_coords = load_poi_coords(poi_coord_pkl)

    region_edges, geo_edges, poi_to_region, num_regions = build_geo_hyperedges(
        poi_coords=poi_coords,
        region_grid_size=args.region_grid_size,
        geo_k=args.geo_k,
        knn_chunk_size=args.knn_chunk_size,
    )

    out = {
        "num_pois": len(poi_coords),
        "num_regions": num_regions,
        "poi_to_region": poi_to_region,
        "region_edges": region_edges,
        "geo_edges": geo_edges,
    }
    out_path = os.path.join(dataset_dir, f"TH_hyperedges_geo_{args.dataset}.pkl")
    save_pkl(out_path, out)
    print("[TH_build_hyperedges_geo] saved:", out_path)


if __name__ == "__main__":
    main()
