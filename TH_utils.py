# coding=utf-8
"""
THGR-Next 通用工具函数。
"""

import os
import pickle
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_poi_coords(poi_coord_pkl: str) -> Dict[int, Tuple[float, float]]:
    obj = load_pkl(poi_coord_pkl)
    out = {}
    for poi_id, value in obj.items():
        out[int(poi_id)] = (float(value[0]), float(value[1]))
    return out


def build_region_ids(poi_coords: Dict[int, Tuple[float, float]], region_grid_size=0.02):
    num_pois = len(poi_coords)
    region_keys = []
    for poi in range(num_pois):
        lat, lon = poi_coords[poi]
        gx = int(np.floor(lat / region_grid_size))
        gy = int(np.floor(lon / region_grid_size))
        region_keys.append((gx, gy))
    uniq = {k: i for i, k in enumerate(sorted(set(region_keys)))}
    poi_to_region = np.array([uniq[k] for k in region_keys], dtype=np.int32)
    return poi_to_region, len(uniq)


def _pairwise_haversine_chunked(lat_rad, lon_rad, start, end):
    lat1 = lat_rad[start:end][:, None]
    lon1 = lon_rad[start:end][:, None]
    lat2 = lat_rad[None, :]
    lon2 = lon_rad[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def build_knn_neighbors(poi_coords: Dict[int, Tuple[float, float]], k=15, chunk_size=512):
    num_pois = len(poi_coords)
    coords = np.zeros((num_pois, 2), dtype=np.float32)
    for i in range(num_pois):
        lat, lon = poi_coords[i]
        coords[i] = [lat, lon]
    lat_rad = np.radians(coords[:, 0].astype(np.float64))
    lon_rad = np.radians(coords[:, 1].astype(np.float64))
    neighbors = np.zeros((num_pois, k), dtype=np.int32)
    neighbor_dists = np.zeros((num_pois, k), dtype=np.float32)

    for start in range(0, num_pois, chunk_size):
        end = min(start + chunk_size, num_pois)
        dist = _pairwise_haversine_chunked(lat_rad, lon_rad, start, end)
        for row in range(end - start):
            dist[row, start + row] = np.inf
        topk_idx = np.argpartition(dist, kth=k, axis=1)[:, :k]
        topk_dist = np.take_along_axis(dist, topk_idx, axis=1)
        ord_idx = np.argsort(topk_dist, axis=1)
        neighbors[start:end] = np.take_along_axis(topk_idx, ord_idx, axis=1)
        neighbor_dists[start:end] = np.take_along_axis(topk_dist, ord_idx, axis=1).astype(np.float32)

    return neighbors, neighbor_dists


def to_torch_sparse(csr_matrix, device):
    import torch
    coo = csr_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(i, v, torch.Size(coo.shape), device=device).coalesce()


def get_user_complete_traj(sessions_dict):
    users_trajs_dict = {}
    users_trajs_lens_dict = {}
    for user_id, sessions in sessions_dict.items():
        traj = []
        for session in sessions:
            traj.extend(session)
        users_trajs_dict[user_id] = traj
        users_trajs_lens_dict[user_id] = len(traj)
    return users_trajs_dict, users_trajs_lens_dict
