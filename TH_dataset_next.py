# coding=utf-8
"""
THGR-Next 数据集：
1) 加载既有 split（train_poi_zero/test_poi_zero）
2) 加载统一异构超图（TH_hypergraph_*.pkl）
3) 提供截断后的序列 batch
"""

import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from TH_hypergraph_builder import build_and_save
from TH_utils import load_pkl, get_user_complete_traj


def _prepare_graph_tensors(graph_obj, device):
    per_type = {}
    types = []
    for e_type, pack in graph_obj["per_type"].items():
        if pack["num_edges"] == 0:
            continue
        types.append(e_type)
        per_type[e_type] = {
            "node_ids": torch.LongTensor(pack["node_ids"]).to(device),
            "edge_ids_local": torch.LongTensor(pack["edge_ids_local"]).to(device),
            "edge_weight": torch.FloatTensor(pack["edge_weight"]).to(device),
            "num_edges": int(pack["num_edges"]),
        }

    return {
        "types": types,
        "node_type_ids": torch.LongTensor(graph_obj["node_type_ids"]).to(device),
        "per_type": per_type,
    }


class THNextDataset(Dataset):
    def __init__(self, data_filename, num_users, num_pois, padding_idx, args, device, graph_obj=None):
        self.data = load_pkl(data_filename)
        self.sessions_dict = self.data[0]
        self.labels_dict = self.data[1]

        self.num_users = num_users
        self.num_pois = num_pois
        self.padding_idx = padding_idx
        self.device = device

        self.users_trajs_dict, self.users_trajs_lens_dict = get_user_complete_traj(self.sessions_dict)

        if graph_obj is None:
            out_path, graph_obj = build_and_save(args.dataset, args)
            self.graph_path = out_path
        else:
            self.graph_path = "loaded-from-memory"

        self.graph_obj = graph_obj
        self.num_cats = int(graph_obj["num_cats"])
        self.num_regions = int(graph_obj["num_regions"])
        self.num_nodes = int(graph_obj["num_nodes"])

        self.poi_to_cat = torch.LongTensor(graph_obj["poi_to_cat"]).to(device)
        self.poi_to_region = torch.LongTensor(graph_obj["poi_to_region"]).to(device)
        self.poi_latlon = torch.FloatTensor(graph_obj["poi_latlon"]).to(device)

        self.graph_tensors = _prepare_graph_tensors(graph_obj, device)

    def __len__(self):
        return self.num_users

    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = self.users_trajs_lens_dict[user_idx]
        label = self.labels_dict[user_idx]
        return {
            "user_idx": torch.tensor(user_idx),
            "user_seq": torch.tensor(user_seq),
            "user_seq_len": torch.tensor(user_seq_len),
            "label": torch.tensor(label),
        }


def th_collate_fn(batch, padding_value, max_seq_len=256):
    batch_user_idx = []
    batch_user_seq = []
    batch_user_seq_len = []
    batch_label = []

    for item in batch:
        seq = item["user_seq"]
        seq_len = int(item["user_seq_len"].item())

        if max_seq_len is not None and seq_len > max_seq_len:
            seq = seq[-max_seq_len:]
            seq_len = max_seq_len

        batch_user_idx.append(item["user_idx"])
        batch_user_seq.append(seq)
        batch_user_seq_len.append(torch.tensor(seq_len, dtype=item["user_seq_len"].dtype))
        batch_label.append(item["label"])

    pad_user_seq = pad_sequence(batch_user_seq, batch_first=True, padding_value=padding_value)
    batch_user_idx = torch.stack(batch_user_idx)
    batch_user_seq_len = torch.stack(batch_user_seq_len)
    batch_label = torch.stack(batch_label)

    last_pos = (batch_user_seq_len - 1).clamp(min=0)
    batch_range = torch.arange(pad_user_seq.size(0))
    last_poi = pad_user_seq[batch_range, last_pos]
    seq_mask = (pad_user_seq != padding_value).long()

    return {
        "user_idx": batch_user_idx,
        "user_seq": pad_user_seq,
        "user_seq_len": batch_user_seq_len,
        "user_seq_mask": seq_mask,
        "label": batch_label,
        "last_poi": last_poi,
    }
