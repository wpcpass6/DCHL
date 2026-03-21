# coding=utf-8
"""
@author: Yantong Lai
"""

import pickle
import numpy as np
from math import radians, cos, sin, asin, sqrt
import scipy.sparse as sp
import torch


def get_unique_seq(sessions_list):
    """从多段 session 中提取去重后的 POI 序列。"""
    # 步骤1：初始化输出列表
    seq_list = []
    # 步骤2：遍历 session 与 poi，按出现顺序去重
    for session in sessions_list:
        for poi in session:
            if poi in seq_list:
                continue
            else:
                seq_list.append(poi)

    return seq_list


def get_unique_seqs_for_sessions(sessions_dict):
    """为每个用户生成去重序列及其长度。"""
    # 步骤1：初始化字典
    seqs_dict = {}
    seqs_lens_dict = {}
    # 步骤2：逐用户处理
    for key, value in sessions_dict.items():
        seqs_dict[key] = get_unique_seq(value)
        seqs_lens_dict[key] = len(get_unique_seq(value))

    return seqs_dict, seqs_lens_dict


def get_seqs_for_sessions(sessions_dict, padding_idx, max_seq_len):
    """将用户 session 展平为序列，并裁剪/补齐到固定长度。"""
    # 步骤1：初始化输出
    seqs_dict = {}
    seqs_lens_dict = {}
    reverse_seqs_dict = {}
    # 步骤2：逐用户拼接 session
    for key, sessions in sessions_dict.items():
        temp = []
        for session in sessions:
            temp.extend(session)
        if len(temp) >= max_seq_len:
            # 步骤3a：超长则截断末尾 max_seq_len
            temp = temp[-max_seq_len:]
            temp_rev = temp[::-1]
            seqs_dict[key] = temp
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = max_seq_len
        else:
            # 步骤3b：不足则 padding
            temp_new = temp + [padding_idx] * (max_seq_len - len(temp))
            temp_rev = temp[::-1] + [padding_idx] * (max_seq_len - len(temp))
            seqs_dict[key] = temp_new
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = len(temp)

    return seqs_dict, reverse_seqs_dict, seqs_lens_dict


def save_list_with_pkl(filename, list_obj):
    """把 Python list 持久化为 pkl 文件。"""
    with open(filename, 'wb') as f:
        pickle.dump(list_obj, f)


def load_list_with_pkl(filename):
    """从 pkl 文件读取 list。"""
    with open(filename, 'rb') as f:
        list_obj = pickle.load(f)

    return list_obj


def save_dict_to_pkl(pkl_filename, dict_pbj):
    """把 Python dict 持久化为 pkl 文件。"""
    with open(pkl_filename, 'wb') as f:
        pickle.dump(dict_pbj, f)


def load_dict_from_pkl(pkl_filename):
    """从 pkl 文件读取 dict。"""
    with open(pkl_filename, 'rb') as f:
        dict_obj = pickle.load(f)

    return dict_obj


def get_num_sessions(sessions_dict):
    """统计总 session 数。"""
    # 步骤1：累加每个用户的 session 数
    num_sessions = 0
    for value in sessions_dict.values():
        num_sessions += len(value)

    return num_sessions


def get_user_complete_traj(sessions_dict):
    """把每个用户的多段 session 拼接为完整轨迹。"""
    # 步骤1：初始化轨迹与长度字典
    users_trajs_dict = {}
    users_trajs_lens_dict = {}
    for userID, sessions in sessions_dict.items():
        # 步骤2：按时间顺序拼接该用户所有 session
        traj = []
        for session in sessions:
            traj.extend(session)
        users_trajs_dict[userID] = traj
        users_trajs_lens_dict[userID] = len(traj)

    return users_trajs_dict, users_trajs_lens_dict


def get_user_reverse_traj(users_trajs_dict):
    """为每个用户轨迹生成反向序列。"""
    # 步骤1：逐用户反转轨迹
    users_rev_trajs_dict = {}
    for userID, traj in users_trajs_dict.items():
        rev_traj = traj[::-1]
        users_rev_trajs_dict[userID] = rev_traj

    return users_rev_trajs_dict


def gen_poi_geo_adj(num_pois, pois_coos_dict, distance_threshold):
    """根据 Haversine 距离阈值构建 POI-POI 地理邻接矩阵。"""
    poi_geo_adj = np.zeros(shape=(num_pois, num_pois))

    # 遍历 POI 两两距离；小于阈值则连边（无向）
    for poi1 in range(num_pois):
        lat1, lon1 = pois_coos_dict[poi1]
        for poi2 in range(poi1, num_pois):
            lat2, lon2 = pois_coos_dict[poi2]
            hav_dist = haversine_distance(lon1, lat1, lon2, lat2)
            if hav_dist <= distance_threshold:
                poi_geo_adj[poi1, poi2] = 1
                poi_geo_adj[poi2, poi1] = 1

    # 步骤3：转成稀疏矩阵格式
    poi_geo_adj = sp.csr_matrix(poi_geo_adj)

    return poi_geo_adj


def process_users_seqs(users_seqs_dict, padding_idx, max_seq_len):
    """对用户序列做截断/补齐，并生成反向序列。"""
    # 步骤1：初始化输出
    processed_seqs_dict = {}
    reverse_seqs_dict = {}
    for key, seq in users_seqs_dict.items():
        if len(seq) >= max_seq_len:
            # 步骤2a：截断
            temp_seq = seq[-max_seq_len:]
            temp_rev_seq = temp_seq[::-1]
        else:
            # 步骤2b：补齐
            temp_seq = seq + [padding_idx] * (max_seq_len - len(seq))
            temp_rev_seq = seq[::-1] + [padding_idx] * (max_seq_len - len(seq))
        processed_seqs_dict[key] = temp_seq
        reverse_seqs_dict[key] = temp_rev_seq

    return processed_seqs_dict, reverse_seqs_dict


def reverse_users_seqs(processed_users_seqs_dict, padding_idx, max_seq_len):
    """对已 padding 的序列按有效长度反转。"""
    # 步骤1：逐用户查找 padding 起点
    reversed_users_seqs_dict = {}
    for key, seq in processed_users_seqs_dict.items():
        for idx in range(len(seq)):
            if seq[idx] == padding_idx:
                actual_seq = seq[:idx]
                reversed_users_seqs_dict[key] = actual_seq[::-1] + [padding_idx] * (max_seq_len - idx)
                break

    return reversed_users_seqs_dict


def gen_users_seqs_masks(users_seqs_dict, padding_idx):
    """根据 padding 位置生成 0/1 mask。"""
    # 步骤1：逐用户构造 mask
    users_seqs_masks_dict = {}
    for key, seq in users_seqs_dict.items():
        temp_seq = []
        for poi in seq:
            if poi != padding_idx:
                temp_seq.append(1)
            else:
                temp_seq.append(0)
        users_seqs_masks_dict[key] = temp_seq

    return users_seqs_masks_dict


def haversine_distance(lon1, lat1, lon2, lat2):
    """计算两经纬度点的球面距离（单位：km）。"""
    # 步骤1：角度转弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # 步骤2：Haversine 公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371

    return c * r


def euclidean_distance(lon1, lat1, lon2, lat2):
    """计算平面欧氏距离（仅作近似或对比）。"""

    return np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


def gen_geo_seqs_adjs_dict(users_seqs_dict, pois_coos_dict, max_seq_len, padding_idx, eta=1, distance_threshold=2.5, distance_type="haversine"):
    """为每个序列构造地理影响邻接矩阵。"""
    # 步骤1：逐用户初始化邻接
    geo_adjs_dict = {}
    for key, seq in users_seqs_dict.items():
        geo_adj = np.zeros(shape=(max_seq_len, max_seq_len))
        actual_seq = []
        for item in seq:
            if item != padding_idx:
                actual_seq.append(item)
        actual_seq_len = len(actual_seq)
        # 步骤2：遍历序列内两两 POI 计算地理影响
        for i in range(actual_seq_len):
            for j in range(i + 1, actual_seq_len):
                l1 = actual_seq[i]
                l2 = actual_seq[j]
                lat1, lon1 = pois_coos_dict[l1]
                lat2, lon2 = pois_coos_dict[l2]
                if distance_type == "haversine":
                    dist = haversine_distance(lon1, lat1, lon2, lat2)
                elif distance_type == "euclidean":
                    dist = euclidean_distance(lon1, lat1, lon2, lat2)
                if 0 < dist <= distance_threshold:
                    # 步骤3：距离越远影响越弱
                    geo_influence = np.exp(-eta * (dist ** 2))
                    geo_adj[i, j] = geo_influence
                    geo_adj[j, i] = geo_influence
        geo_adjs_dict[key] = geo_adj

    return geo_adjs_dict


def create_user_poi_adj(users_seqs_dict, num_users, num_pois):
    """创建 User-POI 交互矩阵 R 及其转置。"""
    # 步骤1：初始化稀疏矩阵
    R = sp.dok_matrix((num_users, num_pois), dtype=np.float)
    # 步骤2：把交互位置置 1
    for userID, seq in users_seqs_dict.items():
        for itemID in seq:
            itemID = itemID - num_users
            R[userID, itemID] = 1

    return R, R.T


def gen_sparse_interaction_matrix(users_seqs_dict, num_users, num_pois):
    """构建二部图邻接矩阵 A（User 与 POI 拼接）。"""
    # 步骤1：先得到 R 与 R^T
    R, R_T = create_user_poi_adj(users_seqs_dict, num_users, num_pois)
    # 步骤2：拼成块矩阵
    A = sp.dok_matrix((num_users + num_pois, num_users + num_pois), dtype=float)
    A[:num_users, num_users:] = R
    A[num_users:, :num_users] = R_T
    A_sparse = A.tocsr()

    return A_sparse


def normalized_adj(adj, is_symmetric=True):
    """归一化邻接矩阵（对称或行归一化）。"""
    if is_symmetric:
        # 步骤1a：对称归一化 D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1/2).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj * d_mat_inv
    else:
        # 步骤1b：行归一化 D^{-1} A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj

    return norm_adj


def normalized_adj_tensor(adj_tensor):
    """对稠密张量邻接做归一化并转稀疏张量。"""
    # Compute the degree matrix
    degree_tensor = torch.diag(torch.sum(adj_tensor, dim=1))

    # inverse degree
    inverse_degree_tensor = torch.inverse(degree_tensor)

    # normalized adjacency
    norm_adj = torch.matmul(inverse_degree_tensor, adj_tensor)

    # convert the normalized adjacency matrix to a sparse tensor
    sparse_norm_adj = torch.sparse.FloatTensor(norm_adj)

    return sparse_norm_adj


def gen_local_graph(adj):
    """为邻接矩阵加自环并归一化。"""
    G = normalized_adj(adj + sp.eye(adj.shape[0]))

    return G


def gen_sparse_H(sessions_dict, num_pois, num_sessions, start_poiID):
    """构建超图关联矩阵 H（POI x Session）。"""
    # 步骤1：初始化 H
    H = np.zeros(shape=(num_pois, num_sessions))
    sess_idx = 0
    for key, sessions in sessions_dict.items():
        # 步骤2：逐 session 标记关联关系
        for session in sessions:
            for poiID in session:
                new_poiID = poiID - start_poiID
                H[new_poiID, sess_idx] = 1
            sess_idx += 1
    assert sess_idx == num_sessions
    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_pois_session(sessions_dict, num_pois, num_sessions):
    """构建 POI-Session 关联矩阵（session 字典版本）。"""
    # 步骤1：初始化 H
    H = np.zeros(shape=(num_pois, num_sessions))
    for sess_idx, session in sessions_dict.items():
        for poi in session:
            H[poi, sess_idx] = 1
    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_user(sessions_dict, num_pois, num_users):
    """构建 POI-User 关联矩阵。"""
    H = np.zeros(shape=(num_pois, num_users))

    for userID, sessions in sessions_dict.items():
        seq = []
        for session in sessions:
            seq.extend(session)
        # 同一用户访问过的 POI 都归入该用户超边
        for poi in seq:
            H[poi, userID] = 1

    H = sp.csr_matrix(H)

    return H


def gen_sparse_directed_H_poi(users_trajs_dict, num_pois):
    """
    构建有向 POI-POI 关联矩阵。
    行表示 source POI，列表示 target POI。
    """
    #初始化一个全0的稠密矩阵L*L
    H = np.zeros(shape=(num_pois, num_pois))

    for userID, traj in users_trajs_dict.items(): #遍历每个用户的完整轨迹
        # 编码全局转移关系。1-2-3 中 1->2 和 1->3 都是关联关系。
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                H[src_poi, tar_poi] = 1
    H = sp.csr_matrix(H)
    return H


def gen_HG_from_sparse_H(H, conv="sym"):
    """由关联矩阵 H 构建标准超图传播矩阵 HG。"""
    # 步骤1：准备超边权重与度矩阵
    n_edge = H.shape[1]
    W = sp.eye(n_edge)

    HW = H.dot(W)
    DV = sp.csr_matrix(HW.sum(axis=1)).astype(float)
    DE = sp.csr_matrix(H.sum(axis=0)).astype(float)
    invDE1 = DE.power(-1)
    invDE1_ = sp.diags(invDE1.toarray()[0])
    HT = H.T

    if conv == "sym":
        # 步骤2a：对称超图卷积形式
        invDV2 = DV.power(n=-1 / 2)
        invDV2_ = sp.diags(invDV2.toarray()[:, 0])
        HG = invDV2_ * H * W * invDE1_ * HT * invDV2_
    elif conv == "asym":
        # 步骤2b：非对称超图卷积形式
        invDV1 = DV.power(-1)
        invDV1_ = sp.diags(invDV1.toarray()[:, 0])
        HG = invDV1_ * H * W * invDE1_ * HT

        # print("invDV1: \n{}".format(invDV1_.toarray()))
        # print("invDE1: \n{}".format(invDE1_.toarray()))
        # print("B-1 * HT: \n{}".format((invDE1_ * HT).toarray()))
        # print("D * H: \n{}".format((invDV1_ * H).toarray()))

    return HG


def get_hyper_deg(incidence_matrix):
    """计算关联矩阵的节点度逆对角阵。"""
    '''
    # incidence_matrix = [num_nodes, num_hyperedges]
    hyper_deg = np.array(incidence_matrix.sum(axis=axis)).squeeze()
    hyper_deg[hyper_deg == 0.] = 1
    hyper_deg = sp.diags(1.0 / hyper_deg)
    '''

    # H  = [num_node, num_edge]
    # DV = [num_node, num_node]
    # DV * H = [num_node, num_edge]

    # HT = [num_edge, num_node]
    # DE = [num_edge, num_edge]
    # DE * HT = [num_edge, num_node]

    # hyper_deg = incidence_matrix.sum(1)
    # inv_hyper_deg = hyper_deg.power(-1)
    # inv_hyper_deg_diag = sp.diags(inv_hyper_deg.toarray()[0])

    # 统计每个节点关联的超边数，行为POI，列为用户。
    # .sum(axis)用来指定求和方向；0表示按列求和（得到每行的和），1表示按行求和（得到每列的和）。这里我们需要统计每个节点关联的超边数，所以应该按行求和，即 axis=1。
    rowsum = np.array(incidence_matrix.sum(1))

    d_inv = np.power(rowsum, -1).flatten()#取倒数，求平均/归一化
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)#将倒数构造成对角矩阵，方便后续乘法操作

    return d_mat_inv


def transform_csr_matrix_to_tensor(csr_matrix):
    """把 scipy CSR 稀疏矩阵转为 torch 稀疏张量。"""
    # 步骤1：CSR -> COO
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    # 步骤2：构造 indices/values 并生成 sparse tensor
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sp_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return sp_tensor


def get_poi_session_freq(num_pois, num_sessions, sessions_dict):
    """统计每个 POI 在各 session 中的出现次数。"""
    poi_sess_freq_matrix = np.zeros(shape=(num_pois, num_sessions))

    # 步骤1：遍历所有 session 进行计数
    sess_idx = 0
    for userID, sessions in sessions_dict.items():
        for session in sessions:
            for poiID in session:
                poi_sess_freq_matrix[poiID, sess_idx] += 1
            sess_idx += 1

    # 步骤2：转稀疏矩阵
    poi_sess_freq_matrix = sp.csr_matrix(poi_sess_freq_matrix)

    return poi_sess_freq_matrix


def get_all_sessions(sessions_dict):
    """把所有 session 收集为 tensor 列表。"""
    # 步骤1：逐用户遍历 session 并转 tensor
    all_sessions = []

    for userID, sessions in sessions_dict.items():
        for session in sessions:
            all_sessions.append(torch.tensor(session))

    return all_sessions


def get_all_users_seqs(users_trajs_dict):
    """把所有用户轨迹收集为 tensor 列表。"""
    # 步骤1：逐用户把轨迹转 tensor
    all_seqs = []
    for userID, traj in users_trajs_dict.items():
        all_seqs.append(torch.tensor(traj))

    return all_seqs


def sparse_adj_tensor_drop_edge(sp_adj, keep_rate):
    """对 torch 稀疏邻接做随机边丢弃。"""
    if keep_rate == 1.0:
        return sp_adj

    # 步骤1：读取稀疏张量的值与索引
    vals = sp_adj._values()
    idxs = sp_adj._indices()
    edgeNum = vals.size()
    # 步骤2：随机采样保留边
    mask = ((torch.rand(edgeNum) + keep_rate).floor()).type(torch.bool)
    # 步骤3：按 keep_rate 重标定权重
    newVals = vals[mask] / keep_rate
    newIdxs = idxs[:, mask]

    return torch.sparse.FloatTensor(newIdxs, newVals, sp_adj.shape)


def csr_matrix_drop_edge(csr_adj_matrix, keep_rate):
    """
    对 scipy CSR 稀疏邻接做随机边丢弃。
    keep_rate: 保留边的比例（0.0-1.0）。1.0 表示不丢弃。
    返回新的 CSR 邻接矩阵。"""
    if keep_rate == 1.0:
        return csr_adj_matrix

    # 步骤1：转 COO 便于按边采样.COO格式会把非零元素存成行索引 (row) 和列索引 (col) 的一一对应关系，这里的每一对索引就代表一条“超边”连接
    coo = csr_adj_matrix.tocoo()
    row = coo.row # 行索引数组，表示每条边的起点节点ID
    col = coo.col # 列索引数组，表示每条边的终点节点ID
    edgeNum = row.shape[0] # 丢弃操作前的边数（非零元素数）

    # 步骤2：生成随机保留掩码
    mask = np.floor(np.random.rand(edgeNum) + keep_rate).astype(np.bool_)

    # 步骤3：根据掩码构建新邻接
    new_row = row[mask]
    new_col = col[mask]
    new_edgeNum = new_row.shape[0]

    # 步骤4：新边权重npone，表示保留的边权重不变（或可根据需要调整）
    new_values = np.ones(new_edgeNum, dtype=np.float)

    drop_adj_matrix = sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)

    return drop_adj_matrix


