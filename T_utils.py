# coding=utf-8
"""
THGR 工具函数集合。

设计目标：
1) 避免 dense 构图导致的内存峰值；
2) 直接构建稀疏超图关联矩阵；
3) 支持图缓存与可复用构图流程。
"""

import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch


def load_list_with_pkl(filename):
    """读取 list 类型 pkl。"""
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_dict_from_pkl(filename):
    """读取 dict 类型 pkl。"""
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_obj_to_pkl(filename, obj):
    """保存任意 Python 对象到 pkl。"""
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def ensure_dir(path):
    """创建目录（若不存在）。"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_user_complete_traj(sessions_dict):
    """将用户多 session 拼成完整轨迹。"""
    users_trajs_dict = {}
    users_trajs_lens_dict = {}
    for user_id, sessions in sessions_dict.items():
        traj = []
        for session in sessions:
            traj.extend(session)
        users_trajs_dict[user_id] = traj
        users_trajs_lens_dict[user_id] = len(traj)
    return users_trajs_dict, users_trajs_lens_dict


def get_user_reverse_traj(users_trajs_dict):
    """反向轨迹。"""
    return {k: v[::-1] for k, v in users_trajs_dict.items()}


def transform_csr_matrix_to_tensor(csr_matrix):
    """scipy csr -> torch sparse tensor。"""
    coo = csr_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(coo.data.astype(np.float32))
    return torch.sparse.FloatTensor(i, v, torch.Size(coo.shape)).coalesce()


def build_region_ids(pois_coos_dict, region_grid_size=0.02):
    """基于固定网格划分 region。"""
    num_pois = len(pois_coos_dict)
    region_keys = []
    for poi in range(num_pois):
        lat, lon = pois_coos_dict[poi]
        gx = int(np.floor(lat / region_grid_size))
        gy = int(np.floor(lon / region_grid_size))
        region_keys.append((gx, gy))

    uniq = {k: i for i, k in enumerate(sorted(set(region_keys)))}
    poi_to_region = np.array([uniq[k] for k in region_keys], dtype=np.int32)
    return poi_to_region, len(uniq)


def build_category_ids(pois_coos_dict, cat_grid_size=0.01):
    """缺少真实 category 时，使用更细网格近似 category。"""
    num_pois = len(pois_coos_dict)
    cat_keys = []
    for poi in range(num_pois):
        lat, lon = pois_coos_dict[poi]
        gx = int(np.floor(lat / cat_grid_size))
        gy = int(np.floor(lon / cat_grid_size))
        cat_keys.append((gx, gy))

    uniq = {k: i for i, k in enumerate(sorted(set(cat_keys)))}
    poi_to_cat = np.array([uniq[k] for k in cat_keys], dtype=np.int32)
    return poi_to_cat, len(uniq)


def build_category_ids_from_obj(cat_obj, num_pois):
    """从外部 category 对象构建 poi_to_cat。

    支持两类输入：
    1) dict: {poi_id: raw_cat_id}
    2) list/ndarray: 下标是 poi_id，值是 raw_cat_id
    """
    if isinstance(cat_obj, dict):
        raw = [cat_obj[i] for i in range(num_pois)]
    else:
        raw = list(cat_obj)
        if len(raw) < num_pois:
            raise ValueError("category object length is smaller than num_pois")
        raw = raw[:num_pois]

    uniq = {c: i for i, c in enumerate(sorted(set(raw)))}
    poi_to_cat = np.array([uniq[c] for c in raw], dtype=np.int32)
    return poi_to_cat, len(uniq)


def _pairwise_haversine_chunked(lat_rad, lon_rad, start, end):
    """计算 [start:end] 到全部点的球面距离（km）。"""
    lat1 = lat_rad[start:end][:, None]
    lon1 = lon_rad[start:end][:, None]
    lat2 = lat_rad[None, :]
    lon2 = lon_rad[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def build_knn_neighbors(pois_coos_dict, k=20, chunk_size=512):
    """按 chunk 计算每个 POI 的 top-k 空间近邻，避免全量内存峰值。"""
    num_pois = len(pois_coos_dict)
    coords = np.zeros((num_pois, 2), dtype=np.float32)
    for i in range(num_pois):
        lat, lon = pois_coos_dict[i]
        coords[i] = [lat, lon]

    lat_rad = np.radians(coords[:, 0].astype(np.float64))
    lon_rad = np.radians(coords[:, 1].astype(np.float64))
    neighbors = np.zeros((num_pois, k), dtype=np.int32)

    for start in range(0, num_pois, chunk_size):
        end = min(start + chunk_size, num_pois)
        dist = _pairwise_haversine_chunked(lat_rad, lon_rad, start, end)
        for row in range(end - start):
            global_idx = start + row
            dist[row, global_idx] = np.inf
        topk_idx = np.argpartition(dist, kth=k, axis=1)[:, :k]
        topk_dist = np.take_along_axis(dist, topk_idx, axis=1)
        ord_idx = np.argsort(topk_dist, axis=1)
        neighbors[start:end] = np.take_along_axis(topk_idx, ord_idx, axis=1)

    return neighbors


def _build_sparse_incidence(num_nodes, num_edges, node_ids, edge_ids):
    """由 (node_id, edge_id) 列表构建稀疏关联矩阵 H。"""
    vals = np.ones(len(node_ids), dtype=np.float32)
    H = sp.coo_matrix((vals, (node_ids, edge_ids)), shape=(num_nodes, num_edges), dtype=np.float32)
    H.sum_duplicates()
    H.data[:] = 1.0
    return H.tocsr()


def build_thgr_incidence_matrices(num_pois, poi_to_cat, num_cats, poi_to_region, num_regions, neighbors):
    """构建三类超边的关联矩阵：category / region / geo。"""
    poi_offset = 0
    cat_offset = num_pois
    reg_offset = num_pois + num_cats
    num_nodes = num_pois + num_cats + num_regions

    # 1) category 超边：{category节点 + 属于该类别的所有POI}
    cat_node_ids = []
    cat_edge_ids = []
    for poi in range(num_pois):
        c = int(poi_to_cat[poi])
        edge_id = c
        cat_node_ids.append(poi_offset + poi)
        cat_edge_ids.append(edge_id)
        cat_node_ids.append(cat_offset + c)
        cat_edge_ids.append(edge_id)
    H_cat = _build_sparse_incidence(num_nodes, num_cats, cat_node_ids, cat_edge_ids)

    # 2) region 超边：{region节点 + 属于该区域的所有POI}
    reg_node_ids = []
    reg_edge_ids = []
    for poi in range(num_pois):
        r = int(poi_to_region[poi])
        edge_id = r
        reg_node_ids.append(poi_offset + poi)
        reg_edge_ids.append(edge_id)
        reg_node_ids.append(reg_offset + r)
        reg_edge_ids.append(edge_id)
    H_reg = _build_sparse_incidence(num_nodes, num_regions, reg_node_ids, reg_edge_ids)

    # 3) local spatial 超边：每个 POI 一条超边，连接 {poi, region(poi), k个空间近邻POI}
    geo_node_ids = []
    geo_edge_ids = []
    for poi in range(num_pois):
        edge_id = poi
        r = int(poi_to_region[poi])
        geo_node_ids.append(poi_offset + poi)
        geo_edge_ids.append(edge_id)
        geo_node_ids.append(reg_offset + r)
        geo_edge_ids.append(edge_id)
        for nb in neighbors[poi]:
            geo_node_ids.append(poi_offset + int(nb))
            geo_edge_ids.append(edge_id)
    H_geo = _build_sparse_incidence(num_nodes, num_pois, geo_node_ids, geo_edge_ids)

    return H_cat, H_reg, H_geo, num_nodes


def incidence_to_propagation(H):
    """H -> (G_en, G_ne)，其中
    G_ne = D_e^-1 H^T, G_en = D_v^-1 H
    """
    # 1) 保证稀疏格式和精度
    H = H.tocsr().astype(np.float32)
    dv = np.asarray(H.sum(axis=1)).reshape(-1)
    de = np.asarray(H.sum(axis=0)).reshape(-1)
    dv[dv == 0.0] = 1.0
    de[de == 0.0] = 1.0

    # 2) 计算度逆并形成传播矩阵
    Dv_inv = sp.diags((1.0 / dv).astype(np.float32))
    De_inv = sp.diags((1.0 / de).astype(np.float32))
    G_en = (Dv_inv @ H).tocsr()
    G_ne = (De_inv @ H.T).tocsr()
    return G_en, G_ne
