# coding=utf-8
"""
从原始 TSMC2014 文本中构建 THGR 所需的真实类别映射文件。

输出：
1) datasets/<dataset>/<dataset>_poi_cat.pkl
   - dict: {poi_id(int): cat_id(str)}
2) datasets/<dataset>/<dataset>_poi_cat_report.pkl
   - dict: 覆盖率、冲突率等统计信息

说明：
- 本脚本不改变现有 train/test split，仅补充类别映射。
- 匹配策略以经纬度为主（优先高精度四舍五入 key），再做近邻兜底。
"""

import argparse
import collections
import os
import pickle
from typing import Dict, List, Tuple


def _load_poi_coords(poi_coord_pkl: str) -> Dict[int, Tuple[float, float]]:
    """读取处理后数据中的 POI 坐标。"""
    with open(poi_coord_pkl, "rb") as f:
        obj = pickle.load(f)

    coords = {}
    for poi_id, value in obj.items():
        lat, lon = float(value[0]), float(value[1])
        coords[int(poi_id)] = (lat, lon)
    return coords


def _build_raw_indices(raw_txt: str, precisions: List[int]):
    """读取原始 txt，建立多精度坐标索引和粗粒度桶索引。"""
    # 不同精度下：coord_key -> Counter(cat_id)
    precise_maps = {p: collections.defaultdict(collections.Counter) for p in precisions}

    # 粗粒度桶（用于兜底近邻匹配）
    # bucket key: (round(lat,4), round(lon,4)) -> list[(lat, lon, cat)]
    bucket_map = collections.defaultdict(list)

    with open(raw_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 原始格式：8列，以 TAB 分隔
            # [user_id, venue_id, venue_cat_id, venue_cat_name, lat, lon, tz, utc_time]
            parts = line.split("\t")
            if len(parts) < 6:
                continue

            cat_id = parts[2]
            lat = float(parts[4])
            lon = float(parts[5])

            for p in precisions:
                key = (round(lat, p), round(lon, p))
                precise_maps[p][key][cat_id] += 1

            bkey = (round(lat, 4), round(lon, 4))
            bucket_map[bkey].append((lat, lon, cat_id))

    return precise_maps, bucket_map


def _choose_major_cat(counter_obj: collections.Counter) -> str:
    """从计数器中选多数类别。"""
    return counter_obj.most_common(1)[0][0]


def _match_single_poi(
    lat: float,
    lon: float,
    precise_maps,
    bucket_map,
    precisions: List[int],
    fallback_eps: float,
):
    """匹配单个 poi 的类别。

    返回：
    - chosen_cat: str 或 None
    - status: "precise" / "fallback" / "unmatched"
    - candidates: 命中的类别计数器（用于冲突统计）
    """
    # 1) 高精度到低精度 key 匹配
    for p in precisions:
        key = (round(lat, p), round(lon, p))
        if key in precise_maps[p]:
            counter_obj = precise_maps[p][key]
            return _choose_major_cat(counter_obj), "precise", counter_obj

    # 2) 粗粒度邻域兜底（3x3 邻格）
    base_bx = round(lat, 4)
    base_by = round(lon, 4)
    pool = []
    for dx in [-0.0001, 0.0, 0.0001]:
        for dy in [-0.0001, 0.0, 0.0001]:
            bkey = (round(base_bx + dx, 4), round(base_by + dy, 4))
            pool.extend(bucket_map.get(bkey, []))

    if pool:
        # 找欧氏近邻（经纬度空间）
        best_dist2 = None
        best_cat = None
        cat_counter = collections.Counter()
        for la, lo, c in pool:
            d2 = (la - lat) * (la - lat) + (lo - lon) * (lo - lon)
            if d2 <= fallback_eps * fallback_eps:
                cat_counter[c] += 1
                if best_dist2 is None or d2 < best_dist2:
                    best_dist2 = d2
                    best_cat = c

        if cat_counter:
            # 兜底时仍用多数票，不用最近邻，以增强鲁棒性
            return _choose_major_cat(cat_counter), "fallback", cat_counter
        if best_cat is not None:
            return best_cat, "fallback", collections.Counter({best_cat: 1})

    return None, "unmatched", collections.Counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["NYC", "TKY"], required=True)
    parser.add_argument("--raw_dir", type=str, default="T_dataset")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--fallback_eps", type=float, default=1e-4, help="经纬度兜底阈值，单位度")
    args = parser.parse_args()

    dataset = args.dataset
    raw_txt = os.path.join(args.raw_dir, f"dataset_TSMC2014_{dataset}.txt")
    poi_coord_pkl = os.path.join(args.data_dir, dataset, f"{dataset}_pois_coos_poi_zero.pkl")
    out_cat_pkl = os.path.join(args.data_dir, dataset, f"{dataset}_poi_cat.pkl")
    out_report_pkl = os.path.join(args.data_dir, dataset, f"{dataset}_poi_cat_report.pkl")

    if not os.path.exists(raw_txt):
        raise FileNotFoundError(f"raw txt not found: {raw_txt}")
    if not os.path.exists(poi_coord_pkl):
        raise FileNotFoundError(f"poi coord pkl not found: {poi_coord_pkl}")

    poi_coords = _load_poi_coords(poi_coord_pkl)
    precisions = [7, 6, 5]
    precise_maps, bucket_map = _build_raw_indices(raw_txt, precisions)

    poi_to_cat = {}
    matched_precise = 0
    matched_fallback = 0
    unmatched = 0
    conflict_cnt = 0

    for poi_id in sorted(poi_coords.keys()):
        lat, lon = poi_coords[poi_id]
        cat, status, counter_obj = _match_single_poi(
            lat=lat,
            lon=lon,
            precise_maps=precise_maps,
            bucket_map=bucket_map,
            precisions=precisions,
            fallback_eps=args.fallback_eps,
        )

        if cat is None:
            # 若确实未匹配，置为 UNKNOWN，保证 THGR 可运行
            cat = "UNKNOWN"
            unmatched += 1
        else:
            if status == "precise":
                matched_precise += 1
            elif status == "fallback":
                matched_fallback += 1

        if len(counter_obj) > 1:
            conflict_cnt += 1

        poi_to_cat[poi_id] = cat

    total = len(poi_to_cat)
    report = {
        "dataset": dataset,
        "total_pois": total,
        "matched_precise": matched_precise,
        "matched_fallback": matched_fallback,
        "unmatched": unmatched,
        "coverage": (total - unmatched) / max(1, total),
        "conflict_pois": conflict_cnt,
        "conflict_ratio": conflict_cnt / max(1, total),
        "num_unique_categories": len(set(poi_to_cat.values())),
        "raw_txt": raw_txt,
        "poi_coord_pkl": poi_coord_pkl,
    }

    with open(out_cat_pkl, "wb") as f:
        pickle.dump(poi_to_cat, f)
    with open(out_report_pkl, "wb") as f:
        pickle.dump(report, f)

    print("[T_build_poi_cat_from_raw] done")
    print(report)


if __name__ == "__main__":
    main()
