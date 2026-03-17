# coding=utf-8
"""
THGR-Next 主模型：
异构超图静态编码 + 因果序列建模 + 动态空间惩罚。
掩码策略：v1（POI-only）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from TH_HGNN import THHGNN


class THNextModel(nn.Module):
    def __init__(self, num_users, num_pois, num_cats, num_regions, args, device):
        super().__init__()
        self.num_users = num_users
        self.num_pois = num_pois
        self.num_cats = num_cats
        self.num_regions = num_regions
        self.num_nodes = num_pois + num_cats + num_regions
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.hidden_dim
        self.padding_idx = num_pois
        self.device = device

        self.alpha = args.alpha_cat
        self.beta = args.beta_region
        self.mask_ratio = args.mask_ratio

        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        self.hgnn = THHGNN(num_nodes=self.num_nodes, dim=self.emb_dim, num_layers=args.num_hg_layers, dropout=args.dropout)

        self.poi_mask_token = nn.Parameter(torch.zeros(1, self.emb_dim))
        nn.init.xavier_uniform_(self.poi_mask_token)
        self.mask_decoder = nn.Linear(self.emb_dim, self.num_pois)

        self.dist_proj = nn.Sequential(
            nn.Linear(1, self.emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 2, self.emb_dim // 2),
        )
        self.pos_embedding = nn.Embedding(args.max_seq_len, self.emb_dim // 2)

        token_dim = self.emb_dim * 4 + self.emb_dim // 2 + self.emb_dim // 2
        self.input_proj = nn.Linear(token_dim, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=args.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=args.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.n_tf_layers)
        self.gamma_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1) * torch.cos(lat2) * (torch.sin(dlon / 2.0) ** 2)
        c = 2.0 * torch.asin(torch.clamp(torch.sqrt(a), max=1.0))
        return 6371.0 * c

    def _build_step_distance(self, seq, poi_latlon):
        bsz, seqlen = seq.shape
        step_dist = torch.zeros((bsz, seqlen, 1), dtype=torch.float32, device=seq.device)
        if seqlen <= 1:
            return step_dist

        prev_seq = seq[:, :-1]
        cur_seq = seq[:, 1:]
        valid = (prev_seq != self.padding_idx) & (cur_seq != self.padding_idx)
        prev_safe = prev_seq.clamp(max=self.num_pois - 1)
        cur_safe = cur_seq.clamp(max=self.num_pois - 1)

        prev_lat = torch.deg2rad(poi_latlon[prev_safe, 0])
        prev_lon = torch.deg2rad(poi_latlon[prev_safe, 1])
        cur_lat = torch.deg2rad(poi_latlon[cur_safe, 0])
        cur_lon = torch.deg2rad(poi_latlon[cur_safe, 1])

        dist = self._haversine(prev_lat, prev_lon, cur_lat, cur_lon) * valid.float()
        step_dist[:, 1:, 0] = dist
        return step_dist

    def _distance_to_all(self, last_poi, poi_latlon):
        safe_last = last_poi.clamp(max=self.num_pois - 1)
        lat1 = torch.deg2rad(poi_latlon[safe_last, 0]).unsqueeze(1)
        lon1 = torch.deg2rad(poi_latlon[safe_last, 1]).unsqueeze(1)
        lat2 = torch.deg2rad(poi_latlon[:, 0]).unsqueeze(0)
        lon2 = torch.deg2rad(poi_latlon[:, 1]).unsqueeze(0)
        dist = self._haversine(lat1, lon1, lat2, lon2)
        invalid = (last_poi == self.padding_idx).unsqueeze(1)
        return torch.where(invalid, torch.zeros_like(dist), dist)

    def _encode_graph_with_mask_v1(self):
        """POI-only mask fill：仅对 POI 节点注入 mask token。"""
        x0 = self.hgnn.node_embedding.weight
        if (not self.training) or self.mask_ratio <= 0:
            node_emb = self.hgnn(graph_tensors=self.graph_tensors, input_node_emb=x0)
            return node_emb, torch.tensor(0.0, device=self.device)

        num_mask = max(1, int(self.num_pois * self.mask_ratio))
        mask_idx = torch.randperm(self.num_pois, device=self.device)[:num_mask]
        x_masked = x0.clone()
        x_masked[mask_idx] = self.poi_mask_token

        node_emb = self.hgnn(graph_tensors=self.graph_tensors, input_node_emb=x_masked)
        logits = self.mask_decoder(node_emb[mask_idx])
        loss_mask = F.cross_entropy(logits, mask_idx)
        return node_emb, loss_mask

    def forward(self, dataset, batch):
        self.graph_tensors = dataset.graph_tensors

        node_emb, loss_mask = self._encode_graph_with_mask_v1()
        poi_emb = node_emb[: self.num_pois]
        cat_emb = node_emb[self.num_pois : self.num_pois + self.num_cats]
        reg_emb = node_emb[self.num_pois + self.num_cats :]

        poi_cat_emb = cat_emb[dataset.poi_to_cat]
        poi_reg_emb = reg_emb[dataset.poi_to_region]

        seq = batch["user_seq"].to(self.device)
        seq_len = batch["user_seq_len"].to(self.device)
        user_idx = batch["user_idx"].to(self.device)
        last_poi = batch["last_poi"].to(self.device)

        bsz, seqlen = seq.shape
        safe_seq = seq.clamp(max=self.num_pois - 1)

        seq_poi = poi_emb[safe_seq]
        seq_cat = poi_cat_emb[safe_seq]
        seq_reg = poi_reg_emb[safe_seq]
        user_tok = self.user_embedding(user_idx).unsqueeze(1).expand(-1, seqlen, -1)

        step_dist = self._build_step_distance(seq, dataset.poi_latlon)
        dist_tok = self.dist_proj(step_dist)
        pos_idx = torch.arange(seqlen, device=self.device).unsqueeze(0).expand(bsz, -1)
        pos_idx = pos_idx.clamp(max=self.pos_embedding.num_embeddings - 1)
        pos_tok = self.pos_embedding(pos_idx)

        token = torch.cat([seq_poi, seq_cat, seq_reg, user_tok, dist_tok, pos_tok], dim=-1)
        token = self.input_proj(token)

        padding_mask = seq.eq(self.padding_idx)
        causal_mask = torch.triu(torch.ones((seqlen, seqlen), device=self.device, dtype=torch.bool), diagonal=1)
        seq_out = self.seq_encoder(token, mask=causal_mask, src_key_padding_mask=padding_mask)

        last_idx = (seq_len - 1).clamp(min=0)
        batch_idx = torch.arange(bsz, device=self.device)
        st = seq_out[batch_idx, last_idx]

        cand_emb = poi_emb + self.alpha * poi_cat_emb + self.beta * poi_reg_emb
        score_sem = st @ cand_emb.T

        gamma_t = F.softplus(self.gamma_proj(st))
        dist_all = self._distance_to_all(last_poi, dataset.poi_latlon)
        score = score_sem - gamma_t * torch.log1p(dist_all)

        return score, loss_mask
