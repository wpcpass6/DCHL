# coding=utf-8
"""
THGR 模型定义。

模块组成：
1) TTypedHypergraphEncoder：静态异构超图编码（P/C/R 节点，cat/reg/geo 三类超边）；
2) 序列编码器：基于 causal Transformer 建模用户轨迹；
3) 候选打分器：语义匹配 + 动态空间惩罚；
4) 自监督：POI mask fill（简化版本）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTypedHypergraphEncoder(nn.Module):
    def __init__(self, num_nodes, emb_dim, num_layers, dropout):
        """类型感知超图编码器。

        输入：节点初始 embedding。
        传播：node->hyperedge->node。
        特点：按超边类型（cat/reg/geo）使用不同线性变换。
        """
        super().__init__()
        self.num_layers = num_layers
        self.node_embedding = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)

        self.type_linear = nn.ModuleDict(
            {
                "cat": nn.Linear(emb_dim, emb_dim, bias=False),
                "reg": nn.Linear(emb_dim, emb_dim, bias=False),
                "geo": nn.Linear(emb_dim, emb_dim, bias=False),
            }
        )
        self.dropout = dropout

    def forward(self, graph_mats):
        """多层传播并做层均值融合。"""
        # 第 0 层为可学习节点 embedding
        x = self.node_embedding.weight
        all_layers = [x]

        for _ in range(self.num_layers):
            # 对三类超边分别传播后求平均
            agg = 0.0
            for t in ["cat", "reg", "geo"]:
                g_ne = graph_mats[t]["G_ne"]
                g_en = graph_mats[t]["G_en"]
                # node -> hyperedge
                edge_emb = torch.sparse.mm(g_ne, x)
                # hyperedge -> node
                node_emb = torch.sparse.mm(g_en, edge_emb)
                agg = agg + self.type_linear[t](node_emb)
            # 残差 + dropout
            x = x + F.dropout(agg / 3.0, p=self.dropout, training=self.training)
            all_layers.append(x)

        # 层均值融合
        return torch.mean(torch.stack(all_layers), dim=0)


class THGR(nn.Module):
    def __init__(self, num_users, num_pois, num_cats, num_regions, args, device):
        """THGR 主模型。"""
        super().__init__()
        self.num_users = num_users
        self.num_pois = num_pois
        self.num_cats = num_cats
        self.num_regions = num_regions
        self.num_nodes = num_pois + num_cats + num_regions
        self.emb_dim = args.emb_dim
        self.device = device
        self.padding_idx = num_pois
        self.alpha = args.alpha_cat
        self.beta = args.beta_region
        self.mask_ratio = args.mask_ratio

        # 用户不进图，仅作为序列 token 的一个特征分量
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        self.encoder = TTypedHypergraphEncoder(
            num_nodes=self.num_nodes,
            emb_dim=self.emb_dim,
            num_layers=args.num_hg_layers,
            dropout=args.dropout,
        )

        # 将“相邻 check-in 位移距离”映射成向量特征
        self.dist_proj = nn.Sequential(
            nn.Linear(1, self.emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 2, self.emb_dim // 2),
        )

        # 位置编码（绝对位置）
        self.pos_embedding = nn.Embedding(args.max_seq_len, self.emb_dim // 2)

        # token = [poi, cat, reg, user, dist, pos]
        token_dim = self.emb_dim * 4 + self.emb_dim // 2 + self.emb_dim // 2
        self.input_proj = nn.Linear(token_dim, args.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim,
            nhead=args.n_heads,
            dim_feedforward=args.hidden_dim * 4,
            dropout=args.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.n_tf_layers)
        # 动态空间敏感度 gamma_t
        self.gamma_proj = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim // 2), nn.ReLU(), nn.Linear(args.hidden_dim // 2, 1))

        # POI mask fill 解码头（分类到 POI id）
        self.mask_decoder = nn.Linear(self.emb_dim, num_pois)

    @staticmethod
    def _haversine_from_pairs(lat1, lon1, lat2, lon2):
        """输入弧度，输出 km。"""
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1) * torch.cos(lat2) * (torch.sin(dlon / 2.0) ** 2)
        c = 2.0 * torch.asin(torch.clamp(torch.sqrt(a), max=1.0))
        return 6371.0 * c

    def _build_step_distance(self, seq, poi_latlon):
        """构造每一步相对上一步 POI 的位移距离特征。"""
        bsz, seqlen = seq.shape
        step_dist = torch.zeros((bsz, seqlen, 1), dtype=torch.float32, device=seq.device)
        if seqlen <= 1:
            return step_dist

        # 取相邻两步 (t-1, t)
        prev_seq = seq[:, :-1]
        cur_seq = seq[:, 1:]
        valid = (prev_seq != self.padding_idx) & (cur_seq != self.padding_idx)

        prev_safe = prev_seq.clamp(max=self.num_pois - 1)
        cur_safe = cur_seq.clamp(max=self.num_pois - 1)

        prev_lat = torch.deg2rad(poi_latlon[prev_safe, 0])
        prev_lon = torch.deg2rad(poi_latlon[prev_safe, 1])
        cur_lat = torch.deg2rad(poi_latlon[cur_safe, 0])
        cur_lon = torch.deg2rad(poi_latlon[cur_safe, 1])

        # 计算距离并掩掉 padding 引入的伪位置
        dist = self._haversine_from_pairs(prev_lat, prev_lon, cur_lat, cur_lon)
        dist = dist * valid.float()
        step_dist[:, 1:, 0] = dist
        return step_dist

    def _distance_to_all_candidates(self, last_poi, poi_latlon):
        """计算 batch 最后位置到所有候选 POI 的距离。"""
        safe_last = last_poi.clamp(max=self.num_pois - 1)
        lat1 = torch.deg2rad(poi_latlon[safe_last, 0]).unsqueeze(1)
        lon1 = torch.deg2rad(poi_latlon[safe_last, 1]).unsqueeze(1)

        lat2 = torch.deg2rad(poi_latlon[:, 0]).unsqueeze(0)
        lon2 = torch.deg2rad(poi_latlon[:, 1]).unsqueeze(0)
        # 广播计算 [B, num_pois]
        dist = self._haversine_from_pairs(lat1, lon1, lat2, lon2)

        invalid = (last_poi == self.padding_idx).unsqueeze(1)
        dist = torch.where(invalid, torch.zeros_like(dist), dist)
        return dist

    def _mask_fill_loss(self, poi_emb):
        """POI mask fill 自监督损失。"""
        if self.mask_ratio <= 0:
            return torch.tensor(0.0, device=poi_emb.device)

        # 随机采样一部分 POI 节点做重建分类
        num_mask = max(1, int(self.num_pois * self.mask_ratio))
        perm = torch.randperm(self.num_pois, device=poi_emb.device)[:num_mask]
        logits = self.mask_decoder(poi_emb[perm])
        return F.cross_entropy(logits, perm)

    def forward(self, dataset, batch):
        """前向：图编码 -> 序列编码 -> 候选打分。"""
        # 1) 静态图编码
        node_emb = self.encoder(dataset.graph_mats)

        poi_emb = node_emb[: self.num_pois]
        cat_emb = node_emb[self.num_pois : self.num_pois + self.num_cats]
        reg_emb = node_emb[self.num_pois + self.num_cats :]

        poi_cat_emb = cat_emb[dataset.poi_to_cat]
        poi_reg_emb = reg_emb[dataset.poi_to_region]

        # 2) 取 batch 输入
        seq = batch["user_seq"].to(self.device)
        seq_len = batch["user_seq_len"].to(self.device)
        user_idx = batch["user_idx"].to(self.device)
        last_poi = batch["last_poi"].to(self.device)

        bsz, seqlen = seq.shape
        safe_seq = seq.clamp(max=self.num_pois - 1)

        # 3) 组装每一步 token 的多源特征
        seq_poi = poi_emb[safe_seq]
        seq_cat = poi_cat_emb[safe_seq]
        seq_reg = poi_reg_emb[safe_seq]

        user_token = self.user_embedding(user_idx).unsqueeze(1).expand(-1, seqlen, -1)
        step_dist = self._build_step_distance(seq, dataset.poi_latlon)
        dist_token = self.dist_proj(step_dist)

        pos_idx = torch.arange(seqlen, device=self.device).unsqueeze(0).expand(bsz, -1)
        pos_idx = pos_idx.clamp(max=self.pos_embedding.num_embeddings - 1)
        pos_token = self.pos_embedding(pos_idx)

        token = torch.cat([seq_poi, seq_cat, seq_reg, user_token, dist_token, pos_token], dim=-1)
        token = self.input_proj(token)

        # 4) causal Transformer 编码（只看历史）
        padding_mask = seq.eq(self.padding_idx)
        causal_mask = torch.triu(torch.ones((seqlen, seqlen), device=self.device, dtype=torch.bool), diagonal=1)
        seq_out = self.seq_encoder(token, mask=causal_mask, src_key_padding_mask=padding_mask)

        last_idx = (seq_len - 1).clamp(min=0)
        batch_idx = torch.arange(bsz, device=self.device)
        st = seq_out[batch_idx, last_idx]

        # 5) 候选静态表示融合：POI + alpha*Category + beta*Region
        cand_emb = poi_emb + self.alpha * poi_cat_emb + self.beta * poi_reg_emb
        score_sem = st @ cand_emb.T

        # 6) 动态空间惩罚：score = semantic - gamma_t * log(1+dist)
        gamma_t = F.softplus(self.gamma_proj(st))
        dist_to_all = self._distance_to_all_candidates(last_poi, dataset.poi_latlon)
        score = score_sem - gamma_t * torch.log1p(dist_to_all)

        loss_mask = self._mask_fill_loss(poi_emb) if self.training else torch.tensor(0.0, device=self.device)
        return score, loss_mask
