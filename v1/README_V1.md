# V1 实验说明
python -m v1.run_v1 --dataset TKY --data_dir datasets/TKY_v1
我把原版和v1版的实验结果发你。我是优先调整v1的参数，还是仅确保实验可进行然后继续v2、v3和v4再进行参数调整？

本目录是对原版 DCHL 的第一阶段改造：

- 新增 **Time-Category 超图视图**
- 新增 **POI Mask Fill 自监督分支**

## 目录说明

- `v1/dataset_v1.py`：V1 数据集与 Time-Category 图构建
- `v1/model_v1.py`：V1 模型
- `v1/run_v1.py`：V1 训练入口
- `scripts/preprocess_tsmc_for_v1.py`：从原始 TSMC2014 数据预处理

## 1. 先做预处理

以 TKY 为例：

```bash
python scripts/preprocess_tsmc_for_v1.py \
  --dataset TKY \
  --raw_file T_dataset/dataset_TSMC2014_TKY.txt \
  --out_dir datasets/TKY_v1 \
  --target_num_users 2173
```

说明：原版 `run.py` 对 TKY 用户数写死为 2173，因此建议显式设置 `--target_num_users 2173`。

## 2. 跑 V1

```bash
python v1/run_v1.py \
  --dataset TKY \
  --data_dir datasets/TKY_v1 \
  --val_ratio 0.1 \
  --early_stop_metric NDCG10 \
  --early_stop_patience 3
```

说明：
- 训练阶段会从训练用户里切分验证集；
- 早停与 best 模型保存都基于验证集指标，不再用 test 选 epoch。

## 3. 用新数据跑“原版 DCHL”基线

建议把重建数据复制到新目录，单独维护。

如果你想让 `run.py --dataset TKY` 直接读取新数据，需临时替换：

- `datasets/TKY/train_poi_zero.txt`
- `datasets/TKY/test_poi_zero.txt`
- `datasets/TKY/TKY_pois_coos_poi_zero.pkl`

并保留原始文件备份，保证可回滚。
