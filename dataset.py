# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

from utils import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class POIDataset(Dataset):
    def __init__(self, data_filename, pois_coos_filename, num_users, num_pois, padding_idx, args, device):
        """完整用户级数据集。

        步骤：
        1) 读取 session 与 label；
        2) 生成用户完整轨迹与反向轨迹；
        3) 构建地理图；
        4) 构建协同超图；
        5) 构建转移有向超图。
        """

        # get all sessions and labels
        # data 文件中保存为 [sessions_dict, labels_dict]
        # sessions_dict: {user_id: [[session1], [session2], ...]}
        # labels_dict:   {user_id: next_poi_label}
        self.data = load_list_with_pkl(data_filename)
        self.sessions_dict = self.data[0]  # poiID starts from 0
        self.labels_dict = self.data[1]
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)

        # ---------- 基础配置 ----------
        self.num_users = num_users
        self.num_pois = num_pois
        # self.num_sessions = get_num_sessions(self.sessions_dict)
        self.padding_idx = padding_idx
        self.distance_threshold = args.distance_threshold
        self.keep_rate = args.keep_rate
        self.device = device

        # ---------- 用户轨迹展开 ----------
        # 把一个用户的多个 session 串接为完整轨迹
        self.users_trajs_dict, self.users_trajs_lens_dict = get_user_complete_traj(self.sessions_dict)
        # 构造反向轨迹（部分序列模型会用到）
        self.users_rev_trajs_dict = get_user_reverse_traj(self.users_trajs_dict)

        # ---------- 地理视图图构建 ----------
        # 1) 用 Haversine 距离 + 阈值构建邻接
        self.poi_geo_adj = gen_poi_geo_adj(num_pois, self.pois_coos_dict, self.distance_threshold)   # csr_matrix
        # 2) 行归一化
        self.poi_geo_graph_matrix = normalized_adj(adj=self.poi_geo_adj, is_symmetric=False)
        # 3) 转为 torch 稀疏张量
        self.poi_geo_graph = transform_csr_matrix_to_tensor(self.poi_geo_graph_matrix).to(device)

        # ---------- 协同视图超图构建 ----------
        # 步骤1：构建 POI-User 超图
        # H_pu: [num_pois, num_users]，若用户访问过该 POI 则为 1
        self.H_pu = gen_sparse_H_user(self.sessions_dict, num_pois, self.num_users)    # [L, U]
        # 步骤2：超边随机丢弃（数据增强）
        self.H_pu = csr_matrix_drop_edge(self.H_pu, args.keep_rate)
        # 步骤3：按节点度归一化，得到用于消息传播的 HG_pu
        self.Deg_H_pu = get_hyper_deg(self.H_pu)    # [L, L]
        self.HG_pu = self.Deg_H_pu * self.H_pu    # [L, U]
        self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(device)

        # 步骤4：得到转置矩阵 HG_up，用于 node->hyperedge 聚合
        self.H_up = self.H_pu.T    # [U, L]
        self.Deg_H_up = get_hyper_deg(self.H_up)    # [U, U]
        self.HG_up = self.Deg_H_up * self.H_up    # [U, L]
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(device)

        # ---------- 轨迹序列张量 ----------
        # 将所有用户完整轨迹整理成 list[tensor]
        # self.all_train_sessions = get_all_sessions(self.sessions_dict)    # list of tensor
        self.all_train_sessions = get_all_users_seqs(self.users_trajs_dict)
        # pad 到同一长度，便于后续 batch 处理
        self.pad_all_train_sessions = pad_sequence(self.all_train_sessions, batch_first=True, padding_value=padding_idx)
        self.pad_all_train_sessions = self.pad_all_train_sessions.to(device)    # [U, MAX_SEQ_LEN]
        self.max_session_len = self.pad_all_train_sessions.size(1)

        # ---------- 转移视图有向超图构建 ----------
        # 步骤1：构建有向 POI->POI 超图
        # H_poi_src[i, j] = 1 表示 i 在某条轨迹中先于 j 出现
        self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, num_pois)    # [L, L]
        # 步骤2：对有向边做随机丢弃（增强）
        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, args.keep_rate_poi)
        # 步骤3：归一化并转 tensor
        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)    # [L, L]
        self.HG_poi_src = self.Deg_H_poi_src * self.H_poi_src    # [L, L]
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.HG_poi_src).to(device)

        # 步骤4：构建目标端矩阵，和 source 端组成两步有向传播
        self.H_poi_tar = self.H_poi_src.T    # [L, L]
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)    # [L, L]
        self.HG_poi_tar = self.Deg_H_poi_tar * self.H_poi_tar    # [L, L]
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.HG_poi_tar).to(device)

    def __len__(self):
        """返回样本数（用户数）。"""
        return self.num_users

    def __getitem__(self, user_idx):
        """按用户索引返回训练样本。

        步骤：
        1) 取用户正向/反向轨迹；
        2) 构造长度与 mask；
        3) 返回 next-POI label。
        """
        # 步骤1：取轨迹
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = self.users_trajs_lens_dict[user_idx]
        # 步骤2：构造有效位置 mask
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = self.users_rev_trajs_dict[user_idx]
        # label 是该用户下一次访问的 POI（分类目标）
        label = self.labels_dict[user_idx]

        # 步骤3：打包成训练样本字典
        sample = {
            "user_idx": torch.tensor(user_idx).to(self.device),
            "user_seq": torch.tensor(user_seq).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(label).to(self.device),
        }

        return sample


class POIPartialDataset(Dataset):
    def __init__(self, full_dataset, user_indices):
        """子集数据集：从完整数据集中按索引抽取样本。"""
        self.data = [full_dataset[i] for i in user_indices]

    def __len__(self):
        """返回子集样本数。"""
        return len(self.data)

    def __getitem__(self, idx):
        """返回子集中的第 idx 个样本。"""
        return self.data[idx]


class POISessionDataset(Dataset):
    def __init__(self, data_filename, label_filename, pois_coos_filename, num_pois, padding_idx, args, device):
        """会话级数据集（按 session 作为样本）。

        与 POIDataset 的区别：这里不按用户聚合，而是直接以 session 为单位。
        """

        # get all sessions and labels
        # self.data = load_list_with_pkl(data_filename)  # data = [sessions_dict, labels_dict]
        # self.sessions_dict = self.data[0]  # poiID starts from 0
        # self.labels_dict = self.data[1]
        self.sessions_dict = load_dict_from_pkl(data_filename)
        self.labels_dict = load_dict_from_pkl(label_filename)
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)
        self.users_trajs_dict = self.sessions_dict

        # ---------- 基础配置 ----------
        # self.num_users = num_users
        self.num_pois = num_pois
        self.num_sessions = len(self.sessions_dict)
        self.padding_idx = padding_idx
        self.distance_threshold = args.distance_threshold
        self.keep_rate = args.keep_rate
        self.device = device

        # ---------- 地理图构建 ----------
        self.poi_geo_adj = gen_poi_geo_adj(num_pois, self.pois_coos_dict, self.distance_threshold)   # csr_matrix
        self.poi_geo_graph_matrix = normalized_adj(adj=self.poi_geo_adj, is_symmetric=False)
        self.poi_geo_graph = transform_csr_matrix_to_tensor(self.poi_geo_graph_matrix).to(device)

        # ---------- 转移有向超图构建 ----------
        self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, num_pois)    # [L, L]
        # drop edge on csr_matrix H_pu
        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, args.keep_rate_poi)
        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)    # [L, L]
        self.HG_poi_src = self.Deg_H_poi_src * self.H_poi_src    # [L, L]
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.HG_poi_src).to(device)

        # 构建 target 端矩阵
        self.H_poi_tar = self.H_poi_src.T    # [L, L]
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)    # [L, L]
        self.HG_poi_tar = self.Deg_H_poi_tar * self.H_poi_tar    # [L, L]
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.HG_poi_tar).to(device)

        # ---------- 协同超图（POI-Session）构建 ----------
        self.H_poi_session = gen_sparse_H_pois_session(self.sessions_dict, num_pois, self.num_sessions)
        self.HG_col = gen_HG_from_sparse_H(self.H_poi_session)
        self.HG_col = transform_csr_matrix_to_tensor(self.HG_col).to(device)

        # ---------- POI-Session 归一化传播矩阵 ----------
        # self.H_pu = gen_sparse_H_user(self.sessions_dict, num_pois, self.num_users)  # [L, U]
        self.H_pu = self.H_poi_session
        # drop edge on csr_matrix H_pu
        # self.H_pu = csr_matrix_drop_edge(self.H_pu, args.keep_rate)

        # get degree of H_pu
        # self.Deg_H_pu = get_hyper_deg(self.H_pu)  # [L, L]
        # normalize poi-user hypergraph
        # self.HG_pu = self.Deg_H_pu * self.H_pu  # [L, U]
        # self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(device)

        # 计算转置矩阵 HG_up: [num_sessions, num_pois]
        self.H_up = self.H_pu.T  # [U, L]
        self.Deg_H_up = get_hyper_deg(self.H_up)  # [U, U]
        self.HG_up = self.Deg_H_up * self.H_up  # [U, L]
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(device)

    def __len__(self):
        """返回 session 数。"""
        # return self.num_users
        return self.num_sessions

    def __getitem__(self, user_idx):
        """按 session 索引返回样本。"""
        # 1) 获取会话序列
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = len(user_seq)
        # 2) 构造 mask 与反向序列
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = user_seq[::-1]
        # 3) 取标签
        label = self.labels_dict[user_idx]

        # 4) 打包输出
        sample = {
            "user_idx": torch.tensor(user_idx).to(self.device),
            "user_seq": torch.tensor(user_seq).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(label).to(self.device),
        }

        return sample


def collate_fn_4sq(batch, padding_value=3835):
    """
    组装一个 batch 并做 padding。

    步骤：
    1) 收集每个样本字段；
    2) 对变长序列做 pad；
    3) 其余字段 stack 成 tensor；
    4) 返回批字典。
    """
    # 步骤1：收集字段
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

    # 步骤2：对序列字段做 padding
    pad_user_seq = pad_sequence(batch_user_seq, batch_first=True, padding_value=padding_value)
    pad_user_rev_seq = pad_sequence(batch_user_rev_seq, batch_first=True, padding_value=padding_value)
    pad_user_seq_mask = pad_sequence(batch_user_seq_mask, batch_first=True, padding_value=0)

    # 步骤3：固定长度字段直接 stack
    batch_user_idx = torch.stack(batch_user_idx)
    batch_user_seq_len = torch.stack(batch_user_seq_len)
    batch_label = torch.stack(batch_label)

    # 步骤4：打包返回
    collate_sample = {
        "user_idx": batch_user_idx,
        "user_seq": pad_user_seq,
        "user_rev_seq": pad_user_rev_seq,
        "user_seq_len": batch_user_seq_len,
        "user_seq_mask": pad_user_seq_mask,
        "label": batch_label,
    }

    return collate_sample

