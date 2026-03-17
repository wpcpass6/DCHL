# coding=utf-8
"""
THGR 数据集定义。

职责：
1) 加载 train/test pkl；
2) 展开用户轨迹（正向/反向）；
3) 构建或读取静态异构超图缓存（P/C/R 节点 + cat/reg/geo 超边）；
4) 提供 batch 所需字段（含 last_poi，用于动态空间惩罚）。
"""

import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from T_utils import (
    load_list_with_pkl,
    load_dict_from_pkl,
    save_obj_to_pkl,
    get_user_complete_traj,
    get_user_reverse_traj,
    build_region_ids,
    build_category_ids,
    build_category_ids_from_obj,
    build_knn_neighbors,
    build_thgr_incidence_matrices,
    incidence_to_propagation,
    transform_csr_matrix_to_tensor,
)


class TPOIDataset(Dataset):
    def __init__(self, data_filename, pois_coos_filename, num_users, num_pois, padding_idx, args, device):
        """用户级样本数据集。"""
        # 1) 读取基础数据
        self.data = load_list_with_pkl(data_filename)
        self.sessions_dict = self.data[0]
        self.labels_dict = self.data[1]
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)

        self.num_users = num_users
        self.num_pois = num_pois
        self.padding_idx = padding_idx
        self.device = device

        # 2) 拼接用户完整轨迹，供序列建模使用
        self.users_trajs_dict, self.users_trajs_lens_dict = get_user_complete_traj(self.sessions_dict)
        self.users_rev_trajs_dict = get_user_reverse_traj(self.users_trajs_dict)

        # 保存 POI 经纬度（后续计算动态空间惩罚使用）
        poi_latlon = torch.zeros((num_pois, 2), dtype=torch.float32)
        for poi in range(num_pois):
            lat, lon = self.pois_coos_dict[poi]
            poi_latlon[poi, 0] = float(lat)
            poi_latlon[poi, 1] = float(lon)
        self.poi_latlon = poi_latlon.to(device)

        # 3) 静态异构超图（P/C/R）
        # 为避免每次启动都重构，优先从缓存读取。
        cache_dir = os.path.dirname(pois_coos_filename)
        # category 来源标签也写入缓存名，避免切换类别源时缓存冲突
        cat_source = "approx"
        cat_file_path = ""
        if hasattr(args, "poi_cat_filename") and args.poi_cat_filename:
            cat_file_path = args.poi_cat_filename
            if not os.path.isabs(cat_file_path):
                cat_file_path = os.path.join(cache_dir, cat_file_path)
            if os.path.exists(cat_file_path):
                cat_source = os.path.basename(cat_file_path).replace(".pkl", "")

        cache_name = (
            f"T_graph_cache_p{num_pois}_rg{args.region_grid_size}_cg{args.cat_grid_size}_"
            f"k{args.geo_k}_cat{cat_source}.pkl"
        )
        cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            # 3.1 命中缓存：直接加载
            cache_obj = load_dict_from_pkl(cache_path)
            poi_to_region = cache_obj["poi_to_region"]
            poi_to_cat = cache_obj["poi_to_cat"]
            num_regions = int(cache_obj["num_regions"])
            num_cats = int(cache_obj["num_cats"])
            H_cat = cache_obj["H_cat"]
            H_reg = cache_obj["H_reg"]
            H_geo = cache_obj["H_geo"]
            num_nodes = int(cache_obj["num_nodes"])
        else:
            # 3.2 未命中缓存：按当前参数重新构图
            poi_to_region, num_regions = build_region_ids(self.pois_coos_dict, args.region_grid_size)

            # 类别优先级：
            # 1) 用户显式传入 --poi_cat_filename
            # 2) 自动探测 datasets/<dataset>/ 下常见类别文件名
            # 3) 若都不存在，则退化为“细网格近似类别”
            detected_cat_file = None
            if cat_file_path and os.path.exists(cat_file_path):
                detected_cat_file = cat_file_path
            else:
                common_names = [
                    "{}_pois_cats_poi_zero.pkl".format(os.path.basename(cache_dir)),
                    "{}_pois_category_poi_zero.pkl".format(os.path.basename(cache_dir)),
                    "{}_poi_cat.pkl".format(os.path.basename(cache_dir)),
                    "poi_cat.pkl",
                ]
                for n in common_names:
                    p = os.path.join(cache_dir, n)
                    if os.path.exists(p):
                        detected_cat_file = p
                        break

            if detected_cat_file is not None:
                cat_obj = load_dict_from_pkl(detected_cat_file)
                poi_to_cat, num_cats = build_category_ids_from_obj(cat_obj, num_pois)
            else:
                poi_to_cat, num_cats = build_category_ids(self.pois_coos_dict, args.cat_grid_size)

            neighbors = build_knn_neighbors(self.pois_coos_dict, k=args.geo_k, chunk_size=args.knn_chunk_size)
            H_cat, H_reg, H_geo, num_nodes = build_thgr_incidence_matrices(
                num_pois, poi_to_cat, num_cats, poi_to_region, num_regions, neighbors
            )
            # 3.3 写缓存，后续训练/推理可复用
            save_obj_to_pkl(
                cache_path,
                {
                    "poi_to_region": poi_to_region,
                    "poi_to_cat": poi_to_cat,
                    "num_regions": num_regions,
                    "num_cats": num_cats,
                    "H_cat": H_cat,
                    "H_reg": H_reg,
                    "H_geo": H_geo,
                    "num_nodes": num_nodes,
                },
            )

        self.num_regions = num_regions
        self.num_cats = num_cats
        self.num_nodes = num_nodes
        self.poi_to_region = torch.LongTensor(poi_to_region).to(device)
        self.poi_to_cat = torch.LongTensor(poi_to_cat).to(device)

        # 4) 把 H 转为传播矩阵（node->edge, edge->node）
        G_en_cat, G_ne_cat = incidence_to_propagation(H_cat)
        G_en_reg, G_ne_reg = incidence_to_propagation(H_reg)
        G_en_geo, G_ne_geo = incidence_to_propagation(H_geo)

        self.graph_mats = {
            "cat": {
                "G_en": transform_csr_matrix_to_tensor(G_en_cat).to(device),
                "G_ne": transform_csr_matrix_to_tensor(G_ne_cat).to(device),
            },
            "reg": {
                "G_en": transform_csr_matrix_to_tensor(G_en_reg).to(device),
                "G_ne": transform_csr_matrix_to_tensor(G_ne_reg).to(device),
            },
            "geo": {
                "G_en": transform_csr_matrix_to_tensor(G_en_geo).to(device),
                "G_ne": transform_csr_matrix_to_tensor(G_ne_geo).to(device),
            },
        }

    def __len__(self):
        """样本数 = 用户数。"""
        return self.num_users

    def __getitem__(self, user_idx):
        """返回单个用户样本。"""
        # 轨迹与标签
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = self.users_trajs_lens_dict[user_idx]
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = self.users_rev_trajs_dict[user_idx]
        label = self.labels_dict[user_idx]

        # 不在这里 to(device)，统一在训练/推理循环中搬运到 GPU
        sample = {
            "user_idx": torch.tensor(user_idx),
            "user_seq": torch.tensor(user_seq),
            "user_rev_seq": torch.tensor(user_rev_seq),
            "user_seq_len": torch.tensor(user_seq_len),
            "user_seq_mask": torch.tensor(user_seq_mask),
            "label": torch.tensor(label),
        }
        return sample


def t_collate_fn(batch, padding_value, max_seq_len=256):
    """组装 batch、截断超长序列并做变长 padding。

    截断策略：仅保留最近 max_seq_len 个 POI（更符合 NextPOI 的近期偏好假设）。
    """
    # 1) 收集字段
    batch_user_idx = []
    batch_user_seq = []
    batch_user_rev_seq = []
    batch_user_seq_len = []
    batch_user_seq_mask = []
    batch_label = []

    for item in batch:
        batch_user_idx.append(item["user_idx"])

        seq = item["user_seq"]
        seq_len = int(item["user_seq_len"].item())

        if max_seq_len is not None and seq_len > max_seq_len:
            # 仅保留最近窗口
            seq = seq[-max_seq_len:]
            seq_len = max_seq_len

        # 反向序列和 mask 基于“截断后的序列”重新构造，保证一致性
        rev_seq = torch.flip(seq, dims=[0])
        seq_mask = torch.ones(seq_len, dtype=item["user_seq_mask"].dtype)

        batch_user_seq.append(seq)
        batch_user_rev_seq.append(rev_seq)
        batch_user_seq_len.append(torch.tensor(seq_len, dtype=item["user_seq_len"].dtype))
        batch_user_seq_mask.append(seq_mask)
        batch_label.append(item["label"])

    # 2) 对序列字段做 padding
    pad_user_seq = pad_sequence(batch_user_seq, batch_first=True, padding_value=padding_value)
    pad_user_rev_seq = pad_sequence(batch_user_rev_seq, batch_first=True, padding_value=padding_value)
    pad_user_seq_mask = pad_sequence(batch_user_seq_mask, batch_first=True, padding_value=0)

    # 3) 其余字段 stack
    batch_user_idx = torch.stack(batch_user_idx)
    batch_user_seq_len = torch.stack(batch_user_seq_len)
    batch_label = torch.stack(batch_label)

    # 4) 计算每条序列最后一个有效 POI，供动态空间惩罚使用
    last_pos = (batch_user_seq_len - 1).clamp(min=0)
    batch_range = torch.arange(pad_user_seq.size(0))
    last_poi = pad_user_seq[batch_range, last_pos]

    # 5) 打包返回
    return {
        "user_idx": batch_user_idx,
        "user_seq": pad_user_seq,
        "user_rev_seq": pad_user_rev_seq,
        "user_seq_len": batch_user_seq_len,
        "user_seq_mask": pad_user_seq_mask,
        "label": batch_label,
        "last_poi": last_poi,
    }
