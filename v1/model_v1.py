# coding=utf-8
"""
V1 模型：
1) 在原 DCHL 三视图基础上增加 Time-Category 视图；
2) 增加 POI Mask Fill 自监督重建损失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewHyperConvLayer(nn.Module):
    def __init__(self, emb_dim, device):
        """单层超图卷积（node->hyperedge->node）。"""
        super(MultiViewHyperConvLayer, self).__init__()
        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        # node -> hyperedge
        msg = torch.sparse.mm(HG_up, pois_embs)
        # hyperedge -> node
        out = torch.sparse.mm(HG_pu, msg)
        return out


class DirectedHyperConvLayer(nn.Module):
    def __init__(self):
        """单层有向超图卷积（用于转移视图）。"""
        super(DirectedHyperConvLayer, self).__init__()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)
        return msg_src


class MultiViewHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, emb_dim, dropout, device):
        """多层超图卷积网络：残差 + 层均值。"""
        super(MultiViewHyperConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim, device)

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            pois_embs = self.mv_hconv_layer(pois_embs, pad_all_train_sessions, HG_up, HG_pu)
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        return torch.mean(torch.stack(final_pois_embs), dim=0)


class DirectedHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, device, dropout=0.3):
        """多层有向超图卷积网络：残差 + 层均值。"""
        super(DirectedHyperConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.di_hconv_layer = DirectedHyperConvLayer()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        return torch.mean(torch.stack(final_pois_embs), dim=0)


class GeoConvNetwork(nn.Module):
    def __init__(self, num_layers, dropout):
        """地理图卷积网络：稀疏邻接传播 + 残差 + 层均值。"""
        super(GeoConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, pois_embs, geo_graph):
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            pois_embs = torch.sparse.mm(geo_graph, pois_embs)
            pois_embs = pois_embs + final_pois_embs[-1]
            final_pois_embs.append(pois_embs)
        return torch.mean(torch.stack(final_pois_embs), dim=0)


class MLPDecoder(nn.Module):
    def __init__(self, emb_dim):
        """用于 POI mask 重建的两层 MLP 解码器。"""
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        return self.fc2(F.elu(self.fc1(x)))


class DCHL_V1(nn.Module):
    def __init__(self, num_users, num_pois, args, device):
        """
        DCHL V1：
        1) 原三视图（协同/地理/转移）+ 新增 Time-Category 视图；
        2) 保留跨视图对比学习；
        3) 新增 POI mask-fill 自监督分支。
        """
        super(DCHL_V1, self).__init__()

        self.num_users = num_users
        self.num_pois = num_pois
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim
        self.ssl_temp = args.temperature
        self.mask_rate = args.mask_rate
        self.mask_alpha = args.mask_alpha

        # 嵌入
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=num_pois)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        # 四视图编码器
        self.mv_hconv_network = MultiViewHyperConvNetwork(args.num_mv_layers, args.emb_dim, 0, device)
        self.geo_conv_network = GeoConvNetwork(args.num_geo_layers, args.dropout)
        self.di_hconv_network = DirectedHyperConvNetwork(args.num_di_layers, device, args.dropout)
        self.tc_hconv_network = MultiViewHyperConvNetwork(args.num_tc_layers, args.emb_dim, 0, device)

        # 用户融合门控
        self.hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.gcn_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.tc_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        # 输入级门控参数（四视图）
        self.w_gate_geo = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_geo = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_tc = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_tc = nn.Parameter(torch.FloatTensor(1, args.emb_dim))

        nn.init.xavier_normal_(self.w_gate_geo.data)
        nn.init.xavier_normal_(self.b_gate_geo.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)
        nn.init.xavier_normal_(self.w_gate_tc.data)
        nn.init.xavier_normal_(self.b_gate_tc.data)

        # mask 分支
        self.mask_poi_token = nn.Parameter(torch.randn(1, args.emb_dim))
        self.mask_decoder = MLPDecoder(args.emb_dim)

    def apply_input_gates(self, poi_base_embs):
        """
        输入级门控：给每个视图生成专属 POI 输入 embedding。

        作用：
        - 在编码前做“软解耦”，减少不同视图语义互相干扰。
        """
        geo_gate_pois_embs = torch.multiply(
            poi_base_embs,
            torch.sigmoid(torch.matmul(poi_base_embs, self.w_gate_geo) + self.b_gate_geo)
        )
        seq_gate_pois_embs = torch.multiply(
            poi_base_embs,
            torch.sigmoid(torch.matmul(poi_base_embs, self.w_gate_seq) + self.b_gate_seq)
        )
        col_gate_pois_embs = torch.multiply(
            poi_base_embs,
            torch.sigmoid(torch.matmul(poi_base_embs, self.w_gate_col) + self.b_gate_col)
        )
        tc_gate_pois_embs = torch.multiply(
            poi_base_embs,
            torch.sigmoid(torch.matmul(poi_base_embs, self.w_gate_tc) + self.b_gate_tc)
        )
        return col_gate_pois_embs, geo_gate_pois_embs, seq_gate_pois_embs, tc_gate_pois_embs

    def cal_loss_infonce(self, emb1, emb2):
        """计算两组表示的 InfoNCE 损失。"""
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        return loss

    def cal_loss_cl_multi(self, emb_list):
        """多视图两两对比损失求和。"""
        norm_embs = [F.normalize(x, p=2, dim=1) for x in emb_list]
        loss = 0.0
        for i in range(len(norm_embs)):
            for j in range(i + 1, len(norm_embs)):
                loss += self.cal_loss_infonce(norm_embs[i], norm_embs[j])
        return loss

    @staticmethod
    def sce_loss(x, y, alpha=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return (1 - (x * y).sum(dim=-1)).pow(alpha).mean()

    def encode_views(self, dataset, batch_user_idx, col_in, geo_in, seq_in, tc_in):
        """
        四视图编码并输出：
        - POI 表示：协同/地理/转移/TC
        - 用户表示：由 HG_up 对各视图 POI 表示聚合得到
        """
        # 四视图 POI 编码
        hg_pois_embs = self.mv_hconv_network(col_in, dataset.pad_all_train_sessions, dataset.HG_up, dataset.HG_pu)
        geo_pois_embs = self.geo_conv_network(geo_in, dataset.poi_geo_graph)
        trans_pois_embs = self.di_hconv_network(seq_in, dataset.HG_poi_src, dataset.HG_poi_tar)
        tc_pois_embs = self.tc_hconv_network(tc_in, dataset.pad_all_train_sessions, dataset.HG_tc_up, dataset.HG_tc_pu)

        # 四视图用户表示（由 POI 聚合）
        hg_users = torch.sparse.mm(dataset.HG_up, hg_pois_embs)[batch_user_idx]
        geo_users = torch.sparse.mm(dataset.HG_up, geo_pois_embs)[batch_user_idx]
        trans_users = torch.sparse.mm(dataset.HG_up, trans_pois_embs)[batch_user_idx]
        tc_users = torch.sparse.mm(dataset.HG_up, tc_pois_embs)[batch_user_idx]

        return (
            hg_pois_embs, geo_pois_embs, trans_pois_embs, tc_pois_embs,
            hg_users, geo_users, trans_users, tc_users,
        )

    def build_mask_loss(self, dataset, batch):
        """
        计算 POI mask-fill 自监督损失。

        实现细节：
        - 只在当前 batch 出现过的 POI 上采样 mask，降低无关噪声；
        - 把被 mask 的 POI embedding 替换为可学习 token；
        - 经四视图编码融合后，用 MLP 重建原始 POI 向量；
        - 使用 SCE 损失约束重建质量。
        """
        if self.mask_rate <= 0:
            return torch.tensor(0.0, device=self.device)

        # 仅在当前 batch 出现过的 POI 上采样 mask，减少噪声与开销
        seq = batch["user_seq"]
        valid = seq[seq < self.num_pois]
        if valid.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        cand = torch.unique(valid)
        if cand.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        n_mask = max(1, int(cand.numel() * self.mask_rate))
        perm = torch.randperm(cand.numel(), device=cand.device)
        mask_idx = cand[perm[:n_mask]]

        base = self.poi_embedding.weight[:-1].clone()
        base[mask_idx] = self.mask_poi_token

        col_in, geo_in, seq_in, tc_in = self.apply_input_gates(base)
        poi_views = self.encode_views(
            dataset=dataset,
            batch_user_idx=batch["user_idx"],
            col_in=col_in,
            geo_in=geo_in,
            seq_in=seq_in,
            tc_in=tc_in,
        )[:4]

        norm_views = [F.normalize(x, p=2, dim=1) for x in poi_views]
        fusion_mask_pois = norm_views[0] + norm_views[1] + norm_views[2] + self.args.beta_tc * norm_views[3]

        decoded = self.mask_decoder(fusion_mask_pois[mask_idx])
        target = self.poi_embedding.weight[:-1][mask_idx].detach()
        return self.sce_loss(decoded, target, alpha=self.mask_alpha)

    def forward(self, dataset, batch):
        """
        前向流程：
        1) 四视图编码；
        2) 计算用户/POI 多视图对比损失；
        3) 用户门控融合 + POI 加权融合，得到推荐 logits；
        4) 计算 POI mask-fill 损失。

        返回：
        - prediction: [batch_size, num_pois]
        - loss_cl_user
        - loss_cl_poi
        - loss_mask
        """
        # 1) 正常推荐主干
        poi_base = self.poi_embedding.weight[:-1]
        col_in, geo_in, seq_in, tc_in = self.apply_input_gates(poi_base)

        (
            hg_pois_embs, geo_pois_embs, trans_pois_embs, tc_pois_embs,
            hg_users, geo_users, trans_users, tc_users,
        ) = self.encode_views(
            dataset=dataset,
            batch_user_idx=batch["user_idx"],
            col_in=col_in,
            geo_in=geo_in,
            seq_in=seq_in,
            tc_in=tc_in,
        )

        # 2) CL 损失
        loss_cl_poi = self.cal_loss_cl_multi([hg_pois_embs, geo_pois_embs, trans_pois_embs, tc_pois_embs])
        loss_cl_user = self.cal_loss_cl_multi([hg_users, geo_users, trans_users, tc_users])

        # 3) 融合预测
        norm_poi_views = [
            F.normalize(hg_pois_embs, p=2, dim=1),
            F.normalize(geo_pois_embs, p=2, dim=1),
            F.normalize(trans_pois_embs, p=2, dim=1),
            F.normalize(tc_pois_embs, p=2, dim=1),
        ]
        norm_user_views = [
            F.normalize(hg_users, p=2, dim=1),
            F.normalize(geo_users, p=2, dim=1),
            F.normalize(trans_users, p=2, dim=1),
            F.normalize(tc_users, p=2, dim=1),
        ]

        coef_c = self.hyper_gate(norm_user_views[0])
        coef_g = self.gcn_gate(norm_user_views[1])
        coef_t = self.trans_gate(norm_user_views[2])
        coef_tc = self.tc_gate(norm_user_views[3])

        fusion_users = (
            coef_c * norm_user_views[0]
            + coef_g * norm_user_views[1]
            + coef_t * norm_user_views[2]
            + coef_tc * norm_user_views[3]
        )
        fusion_pois = (
            norm_poi_views[0]
            + norm_poi_views[1]
            + norm_poi_views[2]
            + self.args.beta_tc * norm_poi_views[3]
        )

        prediction = fusion_users @ fusion_pois.T

        # 4) POI mask fill 损失
        loss_mask = self.build_mask_loss(dataset, batch)

        return prediction, loss_cl_user, loss_cl_poi, loss_mask
