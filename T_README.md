# T_THGR 使用说明（详细）

本文档说明 `T_` 前缀代码的设计目标、执行流程和运行方式。

## 1. 文件结构

- `T_utils.py`：构图与稀疏矩阵工具函数。
- `T_dataset.py`：数据集与 batch 组装逻辑。
- `T_model.py`：THGR 主模型（图编码 + 序列编码 + 动态空间打分）。
- `T_run.py`：训练入口。
- `T_inference.py`：推理/评估入口。
- `T_build_poi_cat_from_raw.py`：从原始 TSMC2014 txt 生成 `poi->category` 映射。
- `T_validate_poi_cat.py`：验证类别映射覆盖率与质量。

## 1.1 关于 pkl 生成说明

当前仓库原始代码没有提供从“原始签到日志”到 `train_poi_zero.txt / test_poi_zero.txt / *_pois_coos_poi_zero.pkl` 的生成脚本。

因此你可以直接复用仓库现有 pkl 运行 THGR，不需要额外生成这些基础文件。

若要使用真实类别（推荐），只需额外生成 `datasets/<dataset>/<dataset>_poi_cat.pkl`。

## 2. THGR 核心思路

### 2.1 节点与超边

图中只保留静态实体节点：

- `P`：POI 节点
- `C`：Category 节点（优先使用真实类别文件；缺失时退化为细网格近似）
- `R`：Region 节点（固定网格）

用户不进图，用户表示由 `user embedding` 提供。

超边类型共 3 类：

1. **Category 超边**：连接 `{category, 属于该 category 的 POI}`
2. **Region 超边**：连接 `{region, 属于该 region 的 POI}`
3. **Local Spatial 超边**：每个 POI 一条超边，连接 `{poi, region(poi), kNN(poi)}`

三种类型的超边，与三个视角有什么区别？


### 2.2 为什么内存更稳

- 构图过程采用稀疏矩阵，不走全量 dense 邻接；
- 空间近邻采用 chunked top-k，避免一次性 `P x P` 常驻内存；
- 构图结果缓存到数据目录，后续训练/推理直接复用。

## 3. 训练执行流程

`python T_run.py --dataset TKY`

训练时主要步骤：

1. 读取数据与参数。
0. （可选）若传入 `--poi_cat_filename`，优先使用真实类别映射文件。
2. `TPOIDataset` 构建（或读取）图缓存：
   - `poi_to_region`
   - `poi_to_cat`（真实类别或近似类别）
   - `H_cat / H_reg / H_geo`
3. 把关联矩阵 `H` 转换为传播矩阵：
   - `G_ne = D_e^-1 H^T`（node->edge）
   - `G_en = D_v^-1 H`（edge->node）
4. 模型前向：
   - 图编码得到 `poi/cat/reg` 静态表示；
   - 组装序列 token：`[g_p, g_c, g_r, e_u, dist_step, pos]`；
   - causal Transformer 输出时刻状态 `s_t`；
   - 候选打分：`semantic - gamma_t * log(1 + distance)`。
5. 损失函数：
   - 主损失：`CrossEntropy(next_poi)`
   - 辅助损失：`POI mask fill`
   - 总损失：`L = L_next + lambda_mask * L_mask`
6. 每轮输出 Recall/NDCG，并按 Recall@5 保存最佳模型。

## 4. 推理执行流程

`python T_inference.py --dataset TKY --saved_model_path T_20260317_101010`

推理步骤：

1. 读取训练目录中的配置 `*_T_args.yaml`（若存在），自动回填关键结构参数。
2. 加载测试集与缓存图。
3. 构建模型并加载 `*_T.pt` 权重。
4. 在测试集计算 `Recall@1/5/10/20` 和 `NDCG@1/5/10/20`。

训练集测试集中一样吗？

可选参数：

- `--eval_active_only`：只评估 `datasets/<dataset>/active_user_dict.pkl` 中用户。

## 5. 常用命令

### 5.0 先构建真实类别映射（推荐）

```bash
python T_build_poi_cat_from_raw.py --dataset TKY
python T_build_poi_cat_from_raw.py --dataset NYC
```

### 5.0.1 验证类别映射质量

```bash
python T_validate_poi_cat.py --dataset TKY
python T_validate_poi_cat.py --dataset NYC
```

### 5.1 TKY 训练

```bash
python T_run.py --dataset TKY --batch_size 128 --geo_k 20 --num_hg_layers 2
```

### 5.1.1 TKY 训练（使用真实类别文件）

```bash
python T_run.py --dataset TKY --poi_cat_filename TKY_poi_cat.pkl --batch_size 128 --geo_k 20
```

### 5.1.2 NYC 训练（使用真实类别文件）

```bash
python T_run.py --dataset NYC --poi_cat_filename NYC_poi_cat.pkl --batch_size 128 --geo_k 15
```

### 5.2 NYC 训练

```bash
python T_run.py --dataset NYC --batch_size 128 --geo_k 15 --num_hg_layers 2
```

### 5.3 TKY 推理（全量用户）

```bash
python T_inference.py --dataset TKY --saved_model_path T_20260317_101010
```

### 5.3.1 TKY 推理（指定真实类别文件）

```bash
python T_inference.py --dataset TKY --saved_model_path T_20260317_101010 --poi_cat_filename TKY_poi_cat.pkl
```

### 5.4 TKY 推理（active 用户）

```bash
python T_inference.py --dataset TKY --saved_model_path T_20260317_101010 --eval_active_only
```

## 6. 关键参数建议

- 图规模：`geo_k=10~30`
- 图层数：`num_hg_layers=1~2`
- 序列层数：`n_tf_layers=2`
- hidden 维度：`hidden_dim=64/128`
- mask 比例：`mask_ratio=0.03~0.1`

若显存吃紧，优先降低：

1. `batch_size`
2. `hidden_dim`
3. `geo_k`
4. `max_seq_len`

## 7. 输出与日志

训练日志目录：`logs/T_时间戳/`

主要文件：

- `log_training.txt`：训练与评估日志
- `<dataset>_T_args.yaml`：训练参数快照
- `<dataset>_T.pt`：最优 checkpoint（按 Recall@5）
- `log_inference.txt`：推理日志（由 `T_inference.py` 生成）

## 8. 注意事项

1. `T_inference.py` 的结构参数必须和训练一致；脚本会优先读取训练目录 yaml 自动对齐。
2. 若你修改了网格参数（`region_grid_size/cat_grid_size`）或 `geo_k`，会生成新的图缓存文件。
3. 类别文件优先级：
   - 显式参数 `--poi_cat_filename`
   - 自动探测（数据目录下常见文件名）
   - 若均不存在，退化为近似类别（细网格）。
4. `--poi_cat_filename` 支持两种 pkl 对象：
   - `dict`：`{poi_id: raw_cat_id}`
   - `list/ndarray`：下标是 `poi_id`，值是 `raw_cat_id`
