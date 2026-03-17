# THGR-Next 说明

## 命名

本方案统一使用 `TH_` 前缀文件，区别于旧版 `T_` 文件。

## 核心设计

- 节点：`P(POI) + C(Category) + R(Region)`
- 超边：`func/category + region + local spatial`（默认）
- mobility 超边：可选，默认关闭 `use_mob=False`
- 传播：`Node -> Hyperedge -> Node`，类型变换 + 注意力回传 + 残差
- 训练：`L = L_next + lambda_mask * L_mask`，掩码策略先用 v1（POI-only）
- 选模：`NDCG@10`

## 文件

- `TH_build_hyperedges_func.py`
- `TH_build_hyperedges_geo.py`
- `TH_build_hyperedges_mob.py`
- `TH_hypergraph_builder.py`
- `TH_HGNN.py`
- `TH_dataset_next.py`
- `TH_model_next.py`
- `TH_run_next.py`
- `TH_inference_next.py`

## 运行

### 1) 训练（默认不启用 mobility）

```bash
python TH_run_next.py --dataset TKY --poi_cat_filename TKY_poi_cat.pkl
```

### 2) 推理

```bash
python TH_inference_next.py --dataset TKY --saved_model_path TH_YYYYMMDD_HHMMSS
```

### 3) 若启用 mobility（可选）

```bash
python TH_run_next.py --dataset TKY --poi_cat_filename TKY_poi_cat.pkl --use_mob --mob_window 5 --mob_min_freq 2
```
