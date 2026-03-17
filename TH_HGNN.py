# coding=utf-8
"""
THGR-Next 的类型感知异构超图传播层。

核心：
1) Node -> Hyperedge：按节点类型变换后做加权聚合
2) Hyperedge -> Node：QK 注意力回传 + 残差
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_sum(src, index, dim_size):
    out = src.new_zeros((dim_size, src.size(-1)))
    out.index_add_(0, index, src)
    return out


def _scatter_count(index, dim_size, device):
    ones = torch.ones(index.size(0), dtype=torch.float32, device=device)
    out = torch.zeros(dim_size, dtype=torch.float32, device=device)
    out.index_add_(0, index, ones)
    return out


def _segment_softmax(values, index, dim_size):
    # values: [M], index: [M]
    max_per = torch.full((dim_size,), -1e9, dtype=values.dtype, device=values.device)
    max_per.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    exp_v = torch.exp(values - max_per[index])
    denom = torch.zeros(dim_size, dtype=values.dtype, device=values.device)
    denom.index_add_(0, index, exp_v)
    return exp_v / (denom[index] + 1e-12)


class THHGNNLayer(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.dropout = dropout

        # 节点类型变换（poi/cat/reg）
        self.w_poi = nn.Linear(dim, dim, bias=False)
        self.w_cat = nn.Linear(dim, dim, bias=False)
        self.w_reg = nn.Linear(dim, dim, bias=False)

        # 超边类型变换（func/region/geo/mob）
        self.w_func = nn.Linear(dim, dim, bias=False)
        self.w_region = nn.Linear(dim, dim, bias=False)
        self.w_geo = nn.Linear(dim, dim, bias=False)
        self.w_mob = nn.Linear(dim, dim, bias=False)

        # QKV for edge->node attention
        self.q_func = nn.Linear(dim, dim, bias=False)
        self.k_func = nn.Linear(dim, dim, bias=False)
        self.v_func = nn.Linear(dim, dim, bias=False)

        self.q_region = nn.Linear(dim, dim, bias=False)
        self.k_region = nn.Linear(dim, dim, bias=False)
        self.v_region = nn.Linear(dim, dim, bias=False)

        self.q_geo = nn.Linear(dim, dim, bias=False)
        self.k_geo = nn.Linear(dim, dim, bias=False)
        self.v_geo = nn.Linear(dim, dim, bias=False)

        self.q_mob = nn.Linear(dim, dim, bias=False)
        self.k_mob = nn.Linear(dim, dim, bias=False)
        self.v_mob = nn.Linear(dim, dim, bias=False)

        self.norm = nn.LayerNorm(dim)

    def _node_type_project(self, x, node_type_ids):
        out = torch.zeros_like(x)
        poi_mask = node_type_ids == 0
        cat_mask = node_type_ids == 1
        reg_mask = node_type_ids == 2
        if poi_mask.any():
            out[poi_mask] = self.w_poi(x[poi_mask])
        if cat_mask.any():
            out[cat_mask] = self.w_cat(x[cat_mask])
        if reg_mask.any():
            out[reg_mask] = self.w_reg(x[reg_mask])
        return out

    def _get_type_modules(self, e_type):
        if e_type == "func":
            return self.w_func, self.q_func, self.k_func, self.v_func
        if e_type == "region":
            return self.w_region, self.q_region, self.k_region, self.v_region
        if e_type == "geo":
            return self.w_geo, self.q_geo, self.k_geo, self.v_geo
        return self.w_mob, self.q_mob, self.k_mob, self.v_mob

    def forward(self, x, graph_tensors):
        """x: [N, d]"""
        n_nodes = x.size(0)
        device = x.device
        h_node = self._node_type_project(x, graph_tensors["node_type_ids"])

        total_update = torch.zeros_like(x)
        used_types = 0

        for e_type in graph_tensors["types"]:
            pack = graph_tensors["per_type"][e_type]
            node_ids = pack["node_ids"]
            edge_ids = pack["edge_ids_local"]
            edge_w = pack["edge_weight"]
            num_e = pack["num_edges"]

            if num_e == 0 or node_ids.numel() == 0:
                continue

            w_edge, q_proj, k_proj, v_proj = self._get_type_modules(e_type)

            # Node -> Edge (mean + edge weight)
            edge_sum = _scatter_sum(h_node[node_ids], edge_ids, dim_size=num_e)
            edge_cnt = _scatter_count(edge_ids, dim_size=num_e, device=device).unsqueeze(1)
            edge_emb = edge_sum / torch.clamp(edge_cnt, min=1.0)
            edge_emb = w_edge(edge_emb) * edge_w.unsqueeze(1)

            # Edge -> Node (attention over incident edges)
            q = q_proj(x[node_ids])
            k = k_proj(edge_emb[edge_ids])
            v = v_proj(edge_emb[edge_ids])
            score = (q * k).sum(dim=1) / math.sqrt(self.dim)
            attn = _segment_softmax(score, node_ids, dim_size=n_nodes).unsqueeze(1)
            msg = v * attn
            node_update = _scatter_sum(msg, node_ids, dim_size=n_nodes)

            total_update = total_update + node_update
            used_types += 1

        if used_types > 0:
            total_update = total_update / float(used_types)

        out = x + F.dropout(total_update, p=self.dropout, training=self.training)
        out = self.norm(out)
        return out


class THHGNN(nn.Module):
    def __init__(self, num_nodes, dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        self.layers = nn.ModuleList([THHGNNLayer(dim=dim, dropout=dropout) for _ in range(num_layers)])

    def forward(self, graph_tensors, input_node_emb=None):
        if input_node_emb is None:
            x = self.node_embedding.weight
        else:
            x = input_node_emb

        all_layers = [x]
        for layer in self.layers:
            x = layer(x, graph_tensors)
            all_layers.append(x)
        return torch.mean(torch.stack(all_layers), dim=0)
