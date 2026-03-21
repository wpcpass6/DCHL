# 推荐目录结构（保留原版不动）

```text
SIGIR2024_DCHL/
  run.py / model.py / dataset.py        # 原版基线（尽量不改）
  v1/                                   # V1 改进代码
  scripts/                              # 数据处理与实验脚本
  docs/                                 # 方案与记录
  datasets/
    TKY/                                # 原版数据（保留）
    TKY_v1/                             # 重建后的新数据
    NYC/
    NYC_v1/
```

实践建议：

1. 原版文件只用于 baseline；
2. 新功能统一放 `v1/`，避免与 baseline 互相污染；
3. 预处理脚本统一放 `scripts/`；
4. 每次跑实验都把参数 yaml 与日志放在独立目录（如 `logs_v1/`）。
