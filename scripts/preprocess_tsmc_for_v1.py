# coding=utf-8
"""
从 TSMC2014 原始数据生成 DCHL/V1 可用数据。

输出文件：
1) train_poi_zero.txt                  -> pickle: [sessions_dict, labels_dict]
2) test_poi_zero.txt                   -> pickle: [sessions_dict, labels_dict]
3) {DATASET}_pois_coos_poi_zero.pkl    -> dict: poi_id -> (lat, lon)
4) {DATASET}_poi_cat.pkl               -> dict: poi_id -> cat_id
5) {DATASET}_cat_name.pkl              -> dict: cat_id -> cat_name
6) train_time_slot.pkl                 -> dict: user_id -> list[list[int]]
7) test_time_slot.pkl                  -> dict: user_id -> list[list[int]]
8) stats.json                          -> 预处理统计信息

说明：
- 会话按“时间间隔阈值”切分（默认24小时）：local_time = utc_time + timezone_offset_minutes。
- 过滤规则遵循论文常见设置：
  a) POI 至少被 min_users_per_poi 个不同用户访问；
  b) 会话长度至少 min_session_len；
  c) 用户会话数至少 min_sessions_per_user。
- 训练/测试划分：每个用户按时间顺序会话前 train_ratio 为训练，后续为测试。
- 标签构造：分别在训练/测试 split 中，取“最后一个交互 POI”作为 label，并从输入会话中移除，避免标签泄漏。
"""

import argparse
import json
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def save_pkl(file_path, obj):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def parse_raw_line(line):
    """解析原始 TSV 的一行记录。"""
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 8:
        return None

    # 8 列定义参见 T_dataset/dataset_TSMC2014_readme.txt
    user_raw = parts[0]
    venue_raw = parts[1]
    cat_raw = parts[2]
    cat_name = parts[3]
    lat = float(parts[4])
    lon = float(parts[5])
    tz_offset_min = int(parts[6])
    utc_time = datetime.strptime(parts[7], "%a %b %d %H:%M:%S +0000 %Y")
    local_time = utc_time + timedelta(minutes=tz_offset_min)

    return {
        "user_raw": user_raw,
        "venue_raw": venue_raw,
        "cat_raw": cat_raw,
        "cat_name": cat_name,
        "lat": lat,
        "lon": lon,
        "utc_time": utc_time,
        "local_time": local_time,
    }


def split_sessions_by_time_gap(user_checkins, max_gap_hours=24):
    """按时间间隔切分会话：相邻 check-in 间隔超过阈值则开启新会话。"""
    sessions = []
    current = []
    prev_time = None
    max_gap = timedelta(hours=max_gap_hours)

    for record in user_checkins:
        cur_time = record["local_time"]
        if prev_time is None or (cur_time - prev_time) <= max_gap:
            current.append(record)
        else:
            sessions.append(current)
            current = [record]
        prev_time = cur_time

    if current:
        sessions.append(current)
    return sessions


def pop_last_interaction_as_label(session_pois, session_slots):
    """在一个 split 内弹出最后一个交互作为标签。"""
    pois = [x[:] for x in session_pois]
    slots = [x[:] for x in session_slots]

    i = len(pois) - 1
    while i >= 0 and len(pois[i]) == 0:
        i -= 1
    if i < 0:
        return None, None, None

    label = pois[i].pop()
    slots[i].pop()

    # 移除尾部空会话
    while pois and len(pois[-1]) == 0:
        pois.pop()
        slots.pop()

    return pois, slots, label


def local_time_to_slot(local_time, slot_mode):
    """把本地时间映射为时间槽。"""
    h = local_time.hour
    m = local_time.minute
    if slot_mode == "24h":
        return h
    # 48h：半小时一个槽
    return h * 2 + (1 if m >= 30 else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["TKY", "NYC"], required=True)
    parser.add_argument("--raw_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--min_users_per_poi", type=int, default=5)
    parser.add_argument("--min_session_len", type=int, default=3)
    parser.add_argument("--min_sessions_per_user", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--slot_mode", type=str, default="24h", choices=["24h", "48h"])
    parser.add_argument("--session_gap_hours", type=float, default=24.0)
    parser.add_argument("--target_num_users", type=int, default=None,
                        help="可选：强制截断到目标用户数（按活跃度从高到低保留）")
    parser.add_argument("--enforce_official_stats", action="store_true")
    args = parser.parse_args()

    raw_file = Path(args.raw_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取原始记录
    records = []
    try:
        # 优先使用 utf-8
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                item = parse_raw_line(line)
                if item is not None:
                    records.append(item)
    except UnicodeDecodeError:
        # 原始数据可能含少量非 utf-8 字符，回退到 latin-1
        records = []
        with open(raw_file, "r", encoding="latin-1") as f:
            for line in f:
                item = parse_raw_line(line)
                if item is not None:
                    records.append(item)

    # 2) 根据“至少5个不同用户访问”过滤 POI
    poi_users = defaultdict(set)
    for r in records:
        poi_users[r["venue_raw"]].add(r["user_raw"])
    kept_pois = {p for p, us in poi_users.items() if len(us) >= args.min_users_per_poi}

    # 3) 按用户收集并按时间排序
    user_checkins = defaultdict(list)
    for r in records:
        if r["venue_raw"] in kept_pois:
            user_checkins[r["user_raw"]].append(r)
    for u in user_checkins:
        user_checkins[u].sort(key=lambda x: x["local_time"])

    # 4) 切分会话并应用长度过滤
    user_sessions = {}
    for u, checkins in user_checkins.items():
        sessions = split_sessions_by_time_gap(checkins, max_gap_hours=args.session_gap_hours)
        sessions = [s for s in sessions if len(s) >= args.min_session_len]
        if len(sessions) >= args.min_sessions_per_user:
            user_sessions[u] = sessions

    # 5) 重编号 user/poi/cat
    user_raw_list = sorted(user_sessions.keys())
    user2id = {u: i for i, u in enumerate(user_raw_list)}

    poi_raw_set = set()
    cat_raw_set = set()
    cat_raw_to_name = {}
    for u in user_raw_list:
        for sess in user_sessions[u]:
            for x in sess:
                poi_raw_set.add(x["venue_raw"])
                cat_raw_set.add(x["cat_raw"])
                cat_raw_to_name[x["cat_raw"]] = x["cat_name"]

    poi_raw_list = sorted(poi_raw_set)
    cat_raw_list = sorted(cat_raw_set)
    poi2id = {p: i for i, p in enumerate(poi_raw_list)}
    cat2id = {c: i for i, c in enumerate(cat_raw_list)}

    # 6) POI 坐标、POI 类别
    poi_coos = {}
    poi_cat = {}
    for u in user_raw_list:
        for sess in user_sessions[u]:
            for x in sess:
                pid = poi2id[x["venue_raw"]]
                if pid not in poi_coos:
                    # 与原 utils.py 保持一致，顺序是 (lat, lon)
                    poi_coos[pid] = (x["lat"], x["lon"])
                if pid not in poi_cat:
                    poi_cat[pid] = cat2id[x["cat_raw"]]

    cat_name = {cat2id[c]: cat_raw_to_name[c] for c in cat2id}

    # 7) 构建 train/test 的 [sessions_dict, labels_dict] 以及时间槽字典
    train_sessions_dict = {}
    train_labels_dict = {}
    test_sessions_dict = {}
    test_labels_dict = {}
    train_slots_dict = {}
    test_slots_dict = {}

    dropped_users = 0
    for user_raw in user_raw_list:
        uid = user2id[user_raw]
        sessions = user_sessions[user_raw]
        n_sessions = len(sessions)

        split_idx = int(n_sessions * args.train_ratio)
        if split_idx <= 0:
            split_idx = 1
        if split_idx >= n_sessions:
            split_idx = n_sessions - 1

        train_raw = sessions[:split_idx]
        test_raw = sessions[split_idx:]

        def convert(sess_list):
            all_p = []
            all_t = []
            for sess in sess_list:
                p = []
                t = []
                for x in sess:
                    p.append(poi2id[x["venue_raw"]])
                    t.append(local_time_to_slot(x["local_time"], args.slot_mode))
                all_p.append(p)
                all_t.append(t)
            return all_p, all_t

        tr_pois, tr_slots = convert(train_raw)
        te_pois, te_slots = convert(test_raw)

        tr_pois, tr_slots, tr_label = pop_last_interaction_as_label(tr_pois, tr_slots)
        te_pois, te_slots, te_label = pop_last_interaction_as_label(te_pois, te_slots)

        if tr_pois is None or te_pois is None or len(tr_pois) == 0 or len(te_pois) == 0:
            dropped_users += 1
            continue

        train_sessions_dict[uid] = tr_pois
        train_labels_dict[uid] = tr_label
        test_sessions_dict[uid] = te_pois
        test_labels_dict[uid] = te_label
        train_slots_dict[uid] = tr_slots
        test_slots_dict[uid] = te_slots

    # 8) 若有用户被移除，重映射为连续 user_id
    if dropped_users > 0:
        kept_uid_old = sorted(train_sessions_dict.keys())
        remap = {old_u: new_u for new_u, old_u in enumerate(kept_uid_old)}

        def remap_dict(d):
            return {remap[k]: v for k, v in d.items() if k in remap}

        train_sessions_dict = remap_dict(train_sessions_dict)
        train_labels_dict = remap_dict(train_labels_dict)
        test_sessions_dict = remap_dict(test_sessions_dict)
        test_labels_dict = remap_dict(test_labels_dict)
        train_slots_dict = remap_dict(train_slots_dict)
        test_slots_dict = remap_dict(test_slots_dict)

    # 8.5) 可选：强制用户数对齐（便于与原版硬编码常量兼容）
    if args.target_num_users is not None and len(train_sessions_dict) > args.target_num_users:
        # 以训练 split 交互数作为活跃度，保留更活跃用户
        user_ids = list(train_sessions_dict.keys())
        user_ids.sort(
            key=lambda u: sum(len(s) for s in train_sessions_dict[u]),
            reverse=True
        )
        kept = set(user_ids[:args.target_num_users])

        def filter_dict(d):
            return {k: v for k, v in d.items() if k in kept}

        train_sessions_dict = filter_dict(train_sessions_dict)
        train_labels_dict = filter_dict(train_labels_dict)
        test_sessions_dict = filter_dict(test_sessions_dict)
        test_labels_dict = filter_dict(test_labels_dict)
        train_slots_dict = filter_dict(train_slots_dict)
        test_slots_dict = filter_dict(test_slots_dict)

        # 重映射 user_id 连续化
        kept_uid_old = sorted(train_sessions_dict.keys())
        remap = {old_u: new_u for new_u, old_u in enumerate(kept_uid_old)}

        def remap_dict(d):
            return {remap[k]: v for k, v in d.items()}

        train_sessions_dict = remap_dict(train_sessions_dict)
        train_labels_dict = remap_dict(train_labels_dict)
        test_sessions_dict = remap_dict(test_sessions_dict)
        test_labels_dict = remap_dict(test_labels_dict)
        train_slots_dict = remap_dict(train_slots_dict)
        test_slots_dict = remap_dict(test_slots_dict)

    # 9) 保存输出
    save_pkl(out_dir / "train_poi_zero.txt", [train_sessions_dict, train_labels_dict])
    save_pkl(out_dir / "test_poi_zero.txt", [test_sessions_dict, test_labels_dict])
    save_pkl(out_dir / f"{args.dataset}_pois_coos_poi_zero.pkl", poi_coos)
    save_pkl(out_dir / f"{args.dataset}_poi_cat.pkl", poi_cat)
    save_pkl(out_dir / f"{args.dataset}_cat_name.pkl", cat_name)
    save_pkl(out_dir / "train_time_slot.pkl", train_slots_dict)
    save_pkl(out_dir / "test_time_slot.pkl", test_slots_dict)

    stats = {
        "dataset": args.dataset,
        "num_users": len(train_sessions_dict),
        "num_pois": len(poi_coos),
        "num_categories": len(cat_name),
        "train_sessions": int(sum(len(v) for v in train_sessions_dict.values())),
        "test_sessions": int(sum(len(v) for v in test_sessions_dict.values())),
        "dropped_users_after_label_pop": int(dropped_users),
        "slot_mode": args.slot_mode,
        "session_gap_hours": args.session_gap_hours,
        "target_num_users": args.target_num_users,
        "rules": {
            "min_users_per_poi": args.min_users_per_poi,
            "min_session_len": args.min_session_len,
            "min_sessions_per_user": args.min_sessions_per_user,
            "train_ratio": args.train_ratio,
        },
    }

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=== 预处理完成 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.enforce_official_stats:
        expected = {
            "NYC": (834, 3835),
            "TKY": (2173, 7038),
        }
        exp_u, exp_p = expected[args.dataset]
        if stats["num_users"] != exp_u or stats["num_pois"] != exp_p:
            raise ValueError(
                f"统计不匹配: users={stats['num_users']} pois={stats['num_pois']}, "
                f"expected users={exp_u} pois={exp_p}."
            )


if __name__ == "__main__":
    main()
