# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiViewHyperConvLayer(nn.Module):
    """
    Multi-view Hypergraph Convolutional Layer
    """

    def __init__(self, emb_dim, device):
        """协同视图单层超图卷积。"""
        super(MultiViewHyperConvLayer, self).__init__()

        # self.fc_seq = nn.Linear(2 * emb_dim, emb_dim, bias=True, device=device)
        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)
        self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        """执行一次 node->hyperedge->node 传播。"""
        # pois_embs = [L, d]
        # H_pu = [L, U]
        # H_up = [U, L]
        # pad_all_train_session = [U, MAX_SESS_LEN]

        # 协同超图卷积：先 POI->User 聚合，再 User->POI 传播
        msg_poi_agg = torch.sparse.mm(HG_up, pois_embs)  # [U, d]

        # 将用户侧聚合消息回传到 POI 侧
        propag_pois_embs = torch.sparse.mm(HG_pu, msg_poi_agg)  # [L, d]
        # propag_pois_embs = self.dropout(propag_pois_embs)

        return propag_pois_embs


class DirectedHyperConvLayer(nn.Module):
    """Directed hypergraph convolutional layer"""

    def __init__(self):
        """转移视图单层有向超图卷积。"""
        super(DirectedHyperConvLayer, self).__init__()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        """执行一次有向传播。"""
        # 有向超图卷积：先聚合到 target，再通过 source 回传
        # 对应论文中的 source->hyperedge->target 的定向传播思想
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)

        return msg_src


class MultiViewHyperConvNetwork(nn.Module):
    """
    Multi-view Hypergraph Convolutional Network
    """

    def __init__(self, num_layers, emb_dim, dropout, device):
        """协同视图多层超图卷积网络。"""
        super(MultiViewHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim, device)
        self.dropout = dropout

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        """多层传播 + 残差 + 层均值汇聚。"""
        # 步骤1：保存第 0 层（初始 embedding）
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            # 步骤2：执行一层超图卷积
            pois_embs = self.mv_hconv_layer(pois_embs, pad_all_train_sessions, HG_up, HG_pu)  # [L, d]
            # add residual connection to alleviate over-smoothing issue
            pois_embs = pois_embs + final_pois_embs[-1]
            # 步骤3：dropout 正则
            pois_embs = F.dropout(pois_embs, self.dropout)
            # 步骤4：保存该层结果
            final_pois_embs.append(pois_embs)
        # 步骤5：对所有层做均值融合
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return final_pois_embs


class DirectedHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, device, dropout=0.3):
        """转移视图多层有向超图卷积网络。"""
        super(DirectedHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.di_hconv_layer = DirectedHyperConvLayer()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        """多层有向传播 + 残差 + 层均值。"""
        # 步骤1：保存初始层
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            # 步骤2：一层有向超图传播
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            # add residual connection
            pois_embs = pois_embs + final_pois_embs[-1]
            # 步骤3：dropout 正则
            pois_embs = F.dropout(pois_embs, self.dropout)
            # 步骤4：保存层输出
            final_pois_embs.append(pois_embs)
        # 步骤5：层均值融合
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return final_pois_embs


class GeoConvNetwork(nn.Module):
    def __init__(self, num_layers, dropout):
        """地理视图图卷积网络。"""
        super(GeoConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, pois_embs, geo_graph):
        """多层地理图传播 + 残差 + 层均值。"""
        # 步骤1：保存初始 embedding
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            # 步骤2：稀疏图乘法传播
            pois_embs = torch.sparse.mm(geo_graph, pois_embs)
            # 步骤3：残差连接
            pois_embs = pois_embs + final_pois_embs[-1]
            # pois_embs = F.dropout(pois_embs, self.dropout)
            # 步骤4：保存层输出
            final_pois_embs.append(pois_embs)
        # 步骤5：层均值融合
        output_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return output_pois_embs


class DCHL(nn.Module):
    def __init__(self, num_users, num_pois, args, device):
        """DCHL 主模型。

        包含三路编码：
        1) 协同超图；2) 地理图；3) 转移有向超图；
        并在输出端做跨视图对比学习与自适应融合。
        """
        super(DCHL, self).__init__()

        # ---------- 基础配置 ----------
        self.num_users = num_users
        self.num_pois = num_pois
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim
        self.ssl_temp = args.temperature

        # ---------- 可学习嵌入 ----------
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=num_pois)

        # ---------- 嵌入初始化 ----------
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        # ---------- 三路编码网络 ----------
        self.mv_hconv_network = MultiViewHyperConvNetwork(args.num_mv_layers, args.emb_dim, 0, device)
        self.geo_conv_network = GeoConvNetwork(args.num_geo_layers, args.dropout)
        self.di_hconv_network = DirectedHyperConvNetwork(args.num_di_layers, device, args.dropout)

        # ---------- 用户融合门控（3个视图） ----------
        self.hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.gcn_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        # 预留门控（未在 forward 中使用）
        self.user_hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.user_gcn_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        # ---------- 预留时序模块参数（当前 forward 未使用） ----------
        self.pos_embeddings = nn.Embedding(1500, self.emb_dim, padding_idx=0)
        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        # ---------- 输入级门控参数（用于三视图解耦） ----------
        self.w_gate_geo = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_geo = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        nn.init.xavier_normal_(self.w_gate_geo.data)
        nn.init.xavier_normal_(self.b_gate_geo.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        # ---------- dropout ----------
        self.dropout = nn.Dropout(args.dropout)

    @staticmethod
    def row_shuffle(embedding):
        """按行随机打乱 embedding（常用于负采样/扰动）。"""
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]

        return corrupted_embedding

    def cal_loss_infonce(self, emb1, emb2):
        """计算一对视图的 InfoNCE 损失。"""
        # InfoNCE: 同一实体在不同视图中的表示为正样本，其余为负样本
        # 步骤1：正样本分数
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        # 步骤2：负样本分数（同 batch 全体）
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        # 步骤3：求平均对比损失
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]

        return loss

    def cal_loss_cl_pois(self, hg_pois_embs, geo_pois_embs, trans_pois_embs):
        """计算 POI 三视图两两对比损失之和。"""
        # projection
        # proj_hg_pois_embs = self.proj_hg(hg_pois_embs)
        # proj_geo_pois_embs = self.proj_geo(geo_pois_embs)
        # proj_trans_pois_embs = self.proj_trans(trans_pois_embs)

        # 步骤1：各视图归一化
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        # 步骤2：三组两两 InfoNCE
        loss_cl_pois = 0.0
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs)

        return loss_cl_pois

    def cal_loss_cl_users(self, hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs):
        """计算用户三视图两两对比损失之和。"""
        # 步骤1：各视图归一化
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # 步骤2：三组两两 InfoNCE
        loss_cl_users = 0.0
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_geo_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_trans_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_geo_batch_users_embs, norm_trans_batch_users_embs)

        return loss_cl_users

    def forward(self, dataset, batch):
        """前向过程：三视图编码 -> 对比学习 -> 融合 -> 全量 POI 打分。"""

        # ---------- 步骤1：输入级三路门控 ----------
        # 对输入 POI embedding 做三种门控，分别供协同/转移/地理视图使用
        geo_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_geo) + self.b_gate_geo))
        seq_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_seq) + self.b_gate_seq))
        col_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_col) + self.b_gate_col))

        # ---------- 步骤2：协同视图编码 ----------
        hg_pois_embs = self.mv_hconv_network(col_gate_pois_embs, dataset.pad_all_train_sessions, dataset.HG_up, dataset.HG_pu)
        # 由 POI 表示聚合得到 user 表示（结构感知）
        hg_structural_users_embs = torch.sparse.mm(dataset.HG_up, hg_pois_embs)  # [U, d]
        hg_batch_users_embs = hg_structural_users_embs[batch["user_idx"]]  # [BS, d]

        # ---------- 步骤3：地理视图编码 ----------
        geo_pois_embs = self.geo_conv_network(geo_gate_pois_embs, dataset.poi_geo_graph)  # [L, d]
        # 地理视图下的 user 表示
        geo_structural_users_embs = torch.sparse.mm(dataset.HG_up, geo_pois_embs)
        geo_batch_users_embs = geo_structural_users_embs[batch["user_idx"]]  # [BS, d]

        # ---------- 步骤4：转移视图编码（有向超图） ----------
        trans_pois_embs = self.di_hconv_network(seq_gate_pois_embs, dataset.HG_poi_src, dataset.HG_poi_tar)
        # 转移视图下的 user 表示
        trans_structural_users_embs = torch.sparse.mm(dataset.HG_up, trans_pois_embs)
        trans_batch_users_embs = trans_structural_users_embs[batch["user_idx"]]  # [BS, d]

        # ---------- 步骤5：跨视图对比学习 ----------
        # POI 与 User 两个粒度都做对比
        loss_cl_poi = self.cal_loss_cl_pois(hg_pois_embs, geo_pois_embs, trans_pois_embs)
        loss_cl_user = self.cal_loss_cl_users(hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs)

        # ---------- 步骤6：用于预测前的归一化 ----------
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # ---------- 步骤7：用户三视图自适应融合 ----------
        hyper_coef = self.hyper_gate(norm_hg_batch_users_embs)
        geo_coef = self.gcn_gate(norm_geo_batch_users_embs)
        trans_coef = self.trans_gate(norm_trans_batch_users_embs)

        # ---------- 步骤8：得到最终用户/POI 表示 ----------
        # 用户侧：门控加权；POI 侧：三视图直接相加
        fusion_batch_users_embs = hyper_coef * norm_hg_batch_users_embs + geo_coef * norm_geo_batch_users_embs + trans_coef * norm_trans_batch_users_embs
        fusion_pois_embs = norm_hg_pois_embs + norm_geo_pois_embs + norm_trans_pois_embs

        # ---------- 步骤9：全量 POI 打分 ----------
        # 每个用户向量与全部 POI 向量做点积，作为分类 logits
        prediction = fusion_batch_users_embs @ fusion_pois_embs.T

        return prediction, loss_cl_user, loss_cl_poi



