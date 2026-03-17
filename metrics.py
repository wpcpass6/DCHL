# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import numpy as np


def hit_k(y_pred, y_true, k):
    """命中率：真实标签是否在 top-k 中。"""
    # 1) 取 top-k 索引
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    # 2) 判断是否命中
    if y_true in y_pred_indices:
        return 1
    else:
        return 0


def ndcg_k(y_pred, y_true, k):
    """NDCG@k：若命中则按排名位置折损。"""
    # 1) 取 top-k 索引
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    # 2) 命中则返回折损增益
    if y_true in y_pred_indices:
        position = y_pred_indices.index(y_true) + 1
        return 1 / np.log2(1 + position)
    else:
        return 0


def mAP_metric(y_true_seq, y_pred_seq, k):
    """mAP 指标（针对 next-POI 的简化定义）。"""
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    # 1) 累加每条样本的 AP
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        # 2) 取 top-k 排序结果
        rec_list = y_pred.argsort()[-k:][::-1]
        # 3) 查找真实标签位置
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    # 4) 取平均
    return rlt / len(y_true_seq)


def MRR_metric(y_true_seq, y_pred_seq):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    # 1) 累加每条样本倒数排名
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        # 2) 全量排序并查找真实标签排名
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
    # 3) 取平均
    return rlt / len(y_true_seq)


def batch_performance(batch_y_pred, batch_y_true, k):
    """计算一个 batch 的 Recall@k 与 NDCG@k。"""
    # 1) 初始化累计量
    batch_size = batch_y_pred.size(0)
    batch_recall = 0
    batch_ndcg = 0
    # 2) 按样本逐个统计
    for idx in range(batch_size):
        hit = hit_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_recall += hit
        ndcg = ndcg_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_ndcg += ndcg

    # 3) 归一化为平均值
    recall = batch_recall / batch_size
    ndcg = batch_ndcg / batch_size

    return recall, ndcg


