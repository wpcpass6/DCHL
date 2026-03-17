# coding=utf-8
"""
统一构建 THGR-Next 异构超图：
节点：P + C + R
超边：func / region / geo / (optional) mob
"""

import argparse
import os
from collections import defaultdict

import numpy as np

from TH_utils import load_poi_coords, load_pkl, save_pkl
from TH_build_hyperedges_geo import build_geo_hyperedges
from TH_build_hyperedges_func import build_function_hyperedges
from TH_build_hyperedges_mob import build_mobility_hyperedges


EDGE_TYPE_VOCAB = {"func": 0, "region": 1, "geo": 2, "mob": 3}
NODE_TYPE_VOCAB = {"poi": 0, "cat": 1, "reg": 2}


def _build_incidence_for_type(num_nodes, edges, node_offsets):
    """返回某类超边的 incidence 索引。

    输出：
    - node_ids: [nnz]
    - edge_local_ids: [nnz]
    - edge_weights: [num_edges_type]
    """
    node_ids = []
    edge_ids = []
    edge_weights = []

    for e_local, e in enumerate(edges):
        e_type = e["edge_type"]
        w = float(e.get("weight", 1.0))
        edge_weights.append(w)

        if e_type == "func":
            c = int(e["cat_id"])
            node_ids.append(node_offsets["cat"] + c)
            edge_ids.append(e_local)
            for p in e["nodes"]:
                node_ids.append(node_offsets["poi"] + int(p))
                edge_ids.append(e_local)

        elif e_type == "region":
            r = int(e["region_id"])
            node_ids.append(node_offsets["reg"] + r)
            edge_ids.append(e_local)
            for p in e["nodes"]:
                node_ids.append(node_offsets["poi"] + int(p))
                edge_ids.append(e_local)

        elif e_type == "geo":
            p = int(e["poi_id"])
            r = int(e["region_id"])
            node_ids.append(node_offsets["poi"] + p)
            edge_ids.append(e_local)
            node_ids.append(node_offsets["reg"] + r)
            edge_ids.append(e_local)
            for nb in e["neighbors"]:
                node_ids.append(node_offsets["poi"] + int(nb))
                edge_ids.append(e_local)

        elif e_type == "mob":
            for p in e["nodes"]:
                node_ids.append(node_offsets["poi"] + int(p))
                edge_ids.append(e_local)

    return np.array(node_ids, dtype=np.int64), np.array(edge_ids, dtype=np.int64), np.array(edge_weights, dtype=np.float32)


def build_hypergraph(dataset, data_dir="datasets", poi_cat_filename="", region_grid_size=0.02, geo_k=15, knn_chunk_size=512,
                     use_func=True, use_region=True, use_geo=True, use_mob=False, mob_window=5, mob_min_freq=2):
    dataset_dir = os.path.join(data_dir, dataset)
    poi_coord_pkl = os.path.join(dataset_dir, f"{dataset}_pois_coos_poi_zero.pkl")
    poi_coords = load_poi_coords(poi_coord_pkl)
    num_pois = len(poi_coords)

    poi_cat_file = poi_cat_filename or f"{dataset}_poi_cat.pkl"
    if not os.path.isabs(poi_cat_file):
        poi_cat_file = os.path.join(dataset_dir, poi_cat_file)

    func_edges, poi_to_cat, num_cats = build_function_hyperedges(num_pois, poi_cat_file)
    region_edges, geo_edges, poi_to_region, num_regions = build_geo_hyperedges(
        poi_coords=poi_coords,
        region_grid_size=region_grid_size,
        geo_k=geo_k,
        knn_chunk_size=knn_chunk_size,
    )

    mob_edges = []
    if use_mob:
        train_pkl = os.path.join(dataset_dir, "train_poi_zero.txt")
        mob_edges = build_mobility_hyperedges(train_pkl, window_size=mob_window, min_freq=mob_min_freq)

    node_offsets = {"poi": 0, "cat": num_pois, "reg": num_pois + num_cats}
    num_nodes = num_pois + num_cats + num_regions

    edge_groups = []
    if use_func:
        edge_groups.append(("func", func_edges))
    if use_region:
        edge_groups.append(("region", region_edges))
    if use_geo:
        edge_groups.append(("geo", geo_edges))
    if use_mob:
        edge_groups.append(("mob", mob_edges))

    # 全局边索引
    global_node_ids = []
    global_edge_ids = []
    edge_type_per_edge = []
    edge_weight_global = []

    # 每类边的局部 incidence（供类型化传播）
    per_type = {}
    global_eid = 0
    for e_type, edges in edge_groups:
        nids, eids_local, w_local = _build_incidence_for_type(num_nodes, edges, node_offsets)
        if len(w_local) == 0:
            continue

        eids_global = eids_local + global_eid
        global_node_ids.append(nids)
        global_edge_ids.append(eids_global)
        edge_type_per_edge.extend([EDGE_TYPE_VOCAB[e_type]] * len(w_local))
        edge_weight_global.extend(w_local.tolist())

        per_type[e_type] = {
            "num_edges": len(w_local),
            "node_ids": nids,
            "edge_ids_local": eids_local,
            "edge_weight": w_local,
        }
        global_eid += len(w_local)

    if len(global_node_ids) == 0:
        raise RuntimeError("No hyperedges were built. Check use_* switches.")

    hyperedge_index = np.vstack([np.concatenate(global_node_ids), np.concatenate(global_edge_ids)]).astype(np.int64)
    num_edges = global_eid

    # node type ids
    node_type_ids = np.zeros(num_nodes, dtype=np.int64)
    node_type_ids[node_offsets["cat"] : node_offsets["reg"]] = NODE_TYPE_VOCAB["cat"]
    node_type_ids[node_offsets["reg"] :] = NODE_TYPE_VOCAB["reg"]

    poi_latlon = np.zeros((num_pois, 2), dtype=np.float32)
    for i in range(num_pois):
        poi_latlon[i] = [poi_coords[i][0], poi_coords[i][1]]

    graph_obj = {
        "dataset": dataset,
        "num_pois": num_pois,
        "num_cats": int(num_cats),
        "num_regions": int(num_regions),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "node_offsets": node_offsets,
        "node_type_vocab": NODE_TYPE_VOCAB,
        "edge_type_vocab": EDGE_TYPE_VOCAB,
        "node_type_ids": node_type_ids,
        "edge_type_per_edge": np.array(edge_type_per_edge, dtype=np.int64),
        "edge_weight": np.array(edge_weight_global, dtype=np.float32),
        "hyperedge_index": hyperedge_index,
        "per_type": per_type,
        "poi_to_cat": np.array(poi_to_cat, dtype=np.int64),
        "poi_to_region": np.array(poi_to_region, dtype=np.int64),
        "poi_latlon": poi_latlon,
        "use_switch": {
            "func": bool(use_func),
            "region": bool(use_region),
            "geo": bool(use_geo),
            "mob": bool(use_mob),
        },
    }
    return graph_obj


def build_and_save(dataset, args):
    graph_obj = build_hypergraph(
        dataset=dataset,
        data_dir=args.data_dir,
        poi_cat_filename=args.poi_cat_filename,
        region_grid_size=args.region_grid_size,
        geo_k=args.geo_k,
        knn_chunk_size=args.knn_chunk_size,
        use_func=args.use_func,
        use_region=args.use_region,
        use_geo=args.use_geo,
        use_mob=args.use_mob,
        mob_window=args.mob_window,
        mob_min_freq=args.mob_min_freq,
    )
    dataset_dir = os.path.join(args.data_dir, dataset)
    file_name = (
        f"TH_hypergraph_{dataset}_f{int(args.use_func)}r{int(args.use_region)}"
        f"g{int(args.use_geo)}m{int(args.use_mob)}_k{args.geo_k}_rg{args.region_grid_size}.pkl"
    )
    out_path = os.path.join(dataset_dir, file_name)
    save_pkl(out_path, graph_obj)
    return out_path, graph_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["NYC", "TKY"], required=True)
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--poi_cat_filename", type=str, default="")
    parser.add_argument("--region_grid_size", type=float, default=0.02)
    parser.add_argument("--geo_k", type=int, default=15)
    parser.add_argument("--knn_chunk_size", type=int, default=512)
    parser.add_argument("--use_func", action="store_true")
    parser.add_argument("--use_region", action="store_true")
    parser.add_argument("--use_geo", action="store_true")
    parser.add_argument("--use_mob", action="store_true")
    parser.add_argument("--mob_window", type=int, default=5)
    parser.add_argument("--mob_min_freq", type=int, default=2)
    args = parser.parse_args()

    # 默认开启 func/region/geo，mob 默认关闭
    if (not args.use_func) and (not args.use_region) and (not args.use_geo) and (not args.use_mob):
        args.use_func = True
        args.use_region = True
        args.use_geo = True
        args.use_mob = False

    out_path, graph_obj = build_and_save(args.dataset, args)
    print("[TH_hypergraph_builder] saved:", out_path)
    print("nodes:", graph_obj["num_nodes"], "edges:", graph_obj["num_edges"], "switch:", graph_obj["use_switch"])


if __name__ == "__main__":
    main()
