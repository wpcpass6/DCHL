# coding=utf-8
"""
V1 数据集：在原 DCHL 数据集基础上增加 Time-Category 超图。
"""

from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import (
    load_list_with_pkl,
    load_dict_from_pkl,
    get_user_complete_traj,
    get_user_reverse_traj,
    gen_poi_geo_adj,
    normalized_adj,
    transform_csr_matrix_to_tensor,
    gen_sparse_H_user,
    csr_matrix_drop_edge,
    get_hyper_deg,
    get_all_users_seqs,
    gen_sparse_directed_H_poi,
)


def build_sparse_h_tc(sessions_dict, time_slot_dict, poi_cat_dict, num_pois, tc_weight_mode="log1p"):
    """
    构建 Time-Category 关联矩阵 H_tc: [num_pois, num_tc_edges]。

    说明：
    - 每个 (time_slot, category) 组合对应一个超边；
    - 若 POI 在该组合出现过，则连接；
    - 可选按出现次数加权（默认 log1p 频次）。
    """
    edge_key2id = {}
    row_idx = []
    col_idx = []
    values = []

    # 统计 (poi, edge) 频次
    pair_cnt = {}

    for user_id, sessions in sessions_dict.items():
        slot_sessions = time_slot_dict[user_id]
        for poi_seq, slot_seq in zip(sessions, slot_sessions):
            for poi, slot in zip(poi_seq, slot_seq):
                cat = poi_cat_dict[poi]
                edge_key = (int(slot), int(cat))
                if edge_key not in edge_key2id:
                    edge_key2id[edge_key] = len(edge_key2id)
                eid = edge_key2id[edge_key]
                pair = (int(poi), int(eid))
                pair_cnt[pair] = pair_cnt.get(pair, 0) + 1

    for (poi, eid), cnt in pair_cnt.items():
        row_idx.append(poi)
        col_idx.append(eid)
        if tc_weight_mode == "binary":
            values.append(1.0)
        else:
            values.append(float(np.log1p(cnt)))

    if len(edge_key2id) == 0:
        # 极端兜底：避免空图
        h_tc = csr_matrix((num_pois, 1), dtype=np.float32)
        return h_tc, edge_key2id

    h_tc = csr_matrix(
        (np.asarray(values, dtype=np.float32),
         (np.asarray(row_idx, dtype=np.int64), np.asarray(col_idx, dtype=np.int64))),
        shape=(num_pois, len(edge_key2id)),
        dtype=np.float32,
    )

    return h_tc, edge_key2id


class POIDatasetV1(Dataset):
    def __init__(self, data_filename, pois_coos_filename, poi_cat_filename,
                 time_slot_filename, num_users, num_pois, padding_idx, args, device):
        """V1 训练数据集。"""
        # ---------- 加载基础数据 ----------
        self.data = load_list_with_pkl(data_filename)
        self.sessions_dict = self.data[0]
        self.labels_dict = self.data[1]
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)
        self.poi_cat_dict = load_dict_from_pkl(poi_cat_filename)
        self.time_slot_dict = load_dict_from_pkl(time_slot_filename)

        self.num_users = num_users
        self.num_pois = num_pois
        self.padding_idx = padding_idx
        self.distance_threshold = args.distance_threshold
        self.keep_rate = args.keep_rate
        self.keep_rate_poi = args.keep_rate_poi
        self.keep_rate_tc = args.keep_rate_tc
        self.tc_weight_mode = args.tc_weight_mode
        self.device = device

        # ---------- 轨迹 ----------
        self.users_trajs_dict, self.users_trajs_lens_dict = get_user_complete_traj(self.sessions_dict)
        self.users_rev_trajs_dict = get_user_reverse_traj(self.users_trajs_dict)

        # ---------- 地理图 ----------
        self.poi_geo_adj = gen_poi_geo_adj(num_pois, self.pois_coos_dict, self.distance_threshold)
        self.poi_geo_graph_matrix = normalized_adj(adj=self.poi_geo_adj, is_symmetric=False)
        self.poi_geo_graph = transform_csr_matrix_to_tensor(self.poi_geo_graph_matrix).to(device)

        # ---------- 协同超图 ----------
        self.H_pu = gen_sparse_H_user(self.sessions_dict, num_pois, self.num_users)
        self.H_pu = csr_matrix_drop_edge(self.H_pu, self.keep_rate)

        self.Deg_H_pu = get_hyper_deg(self.H_pu)
        self.HG_pu = transform_csr_matrix_to_tensor(self.Deg_H_pu * self.H_pu).to(device)

        self.H_up = self.H_pu.T
        self.Deg_H_up = get_hyper_deg(self.H_up)
        self.HG_up = transform_csr_matrix_to_tensor(self.Deg_H_up * self.H_up).to(device)

        # ---------- 转移有向超图 ----------
        self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, num_pois)
        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, self.keep_rate_poi)

        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.Deg_H_poi_src * self.H_poi_src).to(device)

        self.H_poi_tar = self.H_poi_src.T
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.Deg_H_poi_tar * self.H_poi_tar).to(device)

        # ---------- Time-Category 超图 ----------
        self.H_tc, self.tc_edge_key2id = build_sparse_h_tc(
            sessions_dict=self.sessions_dict,
            time_slot_dict=self.time_slot_dict,
            poi_cat_dict=self.poi_cat_dict,
            num_pois=self.num_pois,
            tc_weight_mode=self.tc_weight_mode,
        )
        self.H_tc = csr_matrix_drop_edge(self.H_tc, self.keep_rate_tc)

        self.Deg_H_tc = get_hyper_deg(self.H_tc)
        self.HG_tc_pu = transform_csr_matrix_to_tensor(self.Deg_H_tc * self.H_tc).to(device)

        self.H_tc_up = self.H_tc.T
        self.Deg_H_tc_up = get_hyper_deg(self.H_tc_up)
        self.HG_tc_up = transform_csr_matrix_to_tensor(self.Deg_H_tc_up * self.H_tc_up).to(device)

        # ---------- 轨迹 padding ----------
        self.all_train_sessions = get_all_users_seqs(self.users_trajs_dict)
        self.pad_all_train_sessions = pad_sequence(self.all_train_sessions, batch_first=True,
                                                   padding_value=padding_idx).to(device)
        self.max_session_len = self.pad_all_train_sessions.size(1)

    def __len__(self):
        return self.num_users

    def __getitem__(self, user_idx):
        """
        按用户索引返回一个样本。

        说明：
        - V1 与原版保持一致，仍是“用户级样本”；
        - 一个用户对应一个标签（该 split 内最后一个交互 POI）。
        """
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = self.users_trajs_lens_dict[user_idx]
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = self.users_rev_trajs_dict[user_idx]
        label = self.labels_dict[user_idx]

        return {
            "user_idx": torch.tensor(user_idx).to(self.device),
            "user_seq": torch.tensor(user_seq).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(label).to(self.device),
        }


class POIPartialDatasetV1(Dataset):
    """
    V1 子集数据集。

    用途：
    - 从完整训练集按用户索引切出 train/val；
    - 避免重复构图，直接复用完整训练集的样本。
    """

    def __init__(self, full_dataset, user_indices):
        self.data = [full_dataset[i] for i in user_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_4sq(batch, padding_value=3835):
    """与原版保持一致的批处理函数。"""
    batch_user_idx = []
    batch_user_seq = []
    batch_user_rev_seq = []
    batch_user_seq_len = []
    batch_user_seq_mask = []
    batch_label = []
    for item in batch:
        batch_user_idx.append(item["user_idx"])
        batch_user_seq_len.append(item["user_seq_len"])
        batch_label.append(item["label"])
        batch_user_seq.append(item["user_seq"])
        batch_user_rev_seq.append(item["user_rev_seq"])
        batch_user_seq_mask.append(item["user_seq_mask"])

    pad_user_seq = pad_sequence(batch_user_seq, batch_first=True, padding_value=padding_value)
    pad_user_rev_seq = pad_sequence(batch_user_rev_seq, batch_first=True, padding_value=padding_value)
    pad_user_seq_mask = pad_sequence(batch_user_seq_mask, batch_first=True, padding_value=0)

    batch_user_idx = torch.stack(batch_user_idx)
    batch_user_seq_len = torch.stack(batch_user_seq_len)
    batch_label = torch.stack(batch_label)

    return {
        "user_idx": batch_user_idx,
        "user_seq": pad_user_seq,
        "user_rev_seq": pad_user_rev_seq,
        "user_seq_len": batch_user_seq_len,
        "user_seq_mask": pad_user_seq_mask,
        "label": batch_label,
    }


def infer_user_poi_num(data_dir, dataset_name):
    """从数据文件自动推断用户数和 POI 数，避免写死常量。"""
    data_dir = Path(data_dir)
    train_data = load_list_with_pkl(str(data_dir / "train_poi_zero.txt"))
    sessions_dict = train_data[0]
    pois_coos = load_dict_from_pkl(str(data_dir / f"{dataset_name}_pois_coos_poi_zero.pkl"))

    num_users = len(sessions_dict)
    num_pois = len(pois_coos)
    return num_users, num_pois
