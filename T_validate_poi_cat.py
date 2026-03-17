# coding=utf-8
"""
验证 THGR 的 poi->category 映射文件质量。

输入：
- datasets/<dataset>/<dataset>_poi_cat.pkl
- datasets/<dataset>/<dataset>_pois_coos_poi_zero.pkl

输出：
- 终端打印统计结果
"""

import argparse
import collections
import os
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["NYC", "TKY"], required=True)
    parser.add_argument("--data_dir", type=str, default="datasets")
    args = parser.parse_args()

    dataset = args.dataset
    cat_pkl = os.path.join(args.data_dir, dataset, f"{dataset}_poi_cat.pkl")
    poi_pkl = os.path.join(args.data_dir, dataset, f"{dataset}_pois_coos_poi_zero.pkl")
    report_pkl = os.path.join(args.data_dir, dataset, f"{dataset}_poi_cat_report.pkl")

    if not os.path.exists(cat_pkl):
        raise FileNotFoundError(f"category mapping not found: {cat_pkl}")
    if not os.path.exists(poi_pkl):
        raise FileNotFoundError(f"poi coord pkl not found: {poi_pkl}")

    with open(cat_pkl, "rb") as f:
        poi_to_cat = pickle.load(f)
    with open(poi_pkl, "rb") as f:
        poi_coords = pickle.load(f)

    total_pois = len(poi_coords)
    mapped = 0
    unknown = 0
    bad_key = 0
    for poi_id in poi_coords.keys():
        if poi_id not in poi_to_cat:
            bad_key += 1
            continue
        mapped += 1
        if str(poi_to_cat[poi_id]) == "UNKNOWN":
            unknown += 1

    counter = collections.Counter(list(poi_to_cat.values()))
    top10 = counter.most_common(10)

    print("[T_validate_poi_cat] dataset:", dataset)
    print("total_pois:", total_pois)
    print("mapped_keys:", mapped)
    print("missing_keys:", bad_key)
    print("unknown_count:", unknown)
    print("coverage:", mapped / max(1, total_pois))
    print("unknown_ratio:", unknown / max(1, total_pois))
    print("num_unique_categories:", len(counter))
    print("top10_categories:", top10)

    if os.path.exists(report_pkl):
        with open(report_pkl, "rb") as f:
            report = pickle.load(f)
        print("build_report:", report)


if __name__ == "__main__":
    main()
