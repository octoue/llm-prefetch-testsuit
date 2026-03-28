#!/usr/bin/env python3
"""
从大规模 JSONL trace 中做分层抽样，生成轻量化且统计特征接近原始分布的数据集。

分层维度:
1) turn
2) timestamp 分桶
3) type
4) input_length 分桶
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


Record = Dict[str, object]
Stratum = Tuple[int, int, str, int]


def iter_jsonl(path: str) -> Iterable[Record]:
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def quantile(sorted_values: List[int], p: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, int((len(sorted_values) - 1) * p)))
    return sorted_values[idx]


def make_time_bin(ts: float, min_ts: float, max_ts: float, bins: int) -> int:
    if max_ts <= min_ts:
        return 0
    v = (ts - min_ts) / (max_ts - min_ts)
    b = int(v * bins)
    if b < 0:
        return 0
    if b >= bins:
        return bins - 1
    return b


def make_value_bin(v: int, cut_points: List[int]) -> int:
    for i, cp in enumerate(cut_points):
        if v <= cp:
            return i
    return len(cut_points)


def allocate_targets(
    stratum_counts: Dict[Stratum, int],
    sample_n: int,
    total_n: int,
) -> Dict[Stratum, int]:
    keys = list(stratum_counts.keys())
    non_empty = len(keys)

    targets = {k: 0 for k in keys}
    if sample_n <= 0 or total_n <= 0 or not keys:
        return targets

    # 仅在分层数量较少时，才保证每层至少 1 条，避免过多稀疏分层导致分布失真。
    base_budget = 0
    if sample_n >= non_empty and non_empty <= max(1, sample_n // 20):
        for k in keys:
            targets[k] = 1
        base_budget = non_empty

    remaining = sample_n - base_budget
    if remaining <= 0:
        return targets

    fractional: List[Tuple[float, Stratum]] = []
    allocated = 0
    for k in keys:
        c = stratum_counts[k]
        exact = remaining * c / total_n
        floor_v = int(math.floor(exact))
        targets[k] += floor_v
        allocated += floor_v
        fractional.append((exact - floor_v, k))

    remainder = remaining - allocated
    if remainder > 0:
        fractional.sort(key=lambda x: x[0], reverse=True)
        for i in range(remainder):
            targets[fractional[i][1]] += 1

    # 安全约束: 不超过该分层原始样本数
    for k, c in stratum_counts.items():
        if targets[k] > c:
            targets[k] = c
    return targets


def distribution(counter: Counter, total: int) -> Dict[str, float]:
    if total <= 0:
        return {}
    return {str(k): v / total for k, v in sorted(counter.items(), key=lambda x: x[0])}


def l1_distance(d1: Dict[str, float], d2: Dict[str, float]) -> float:
    keys = set(d1.keys()) | set(d2.keys())
    return sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys)


def main() -> int:
    parser = argparse.ArgumentParser(description="按 turn+时间分桶+type 分层采样 JSONL trace")
    parser.add_argument("--input", required=True, help="输入 JSONL 路径")
    parser.add_argument("--output", required=True, help="输出轻量化 JSONL 路径")
    parser.add_argument("--report", required=True, help="输出统计对比报告 (markdown)")
    parser.add_argument("--sample-size", type=int, default=0, help="目标采样条数，0 表示按比例+下限自动计算")
    parser.add_argument("--sample-ratio", type=float, default=0.28, help="自动采样比例")
    parser.add_argument("--min-sample-size", type=int, default=12000, help="自动采样时的最小条数")
    parser.add_argument("--time-bins", type=int, default=10, help="timestamp 分桶数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--input-cut-quantiles",
        default="0.5,0.9,0.99",
        help="input_length 分桶分位点，逗号分隔",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Pass 1: 全量基础统计 + timestamp 范围
    total_n = 0
    min_ts = float("inf")
    max_ts = float("-inf")
    full_turn = Counter()
    full_type = Counter()
    full_time_bin = Counter()
    full_input_vals: List[int] = []
    full_output_vals: List[int] = []
    input_sum = 0
    output_sum = 0

    for rec in iter_jsonl(args.input):
        total_n += 1
        ts = float(rec.get("timestamp", 0.0))
        min_ts = min(min_ts, ts)
        max_ts = max(max_ts, ts)
        t = int(rec.get("turn", 0))
        tp = str(rec.get("type", "unknown"))
        full_turn[t] += 1
        full_type[tp] += 1
        iv = int(rec.get("input_length", 0))
        ov = int(rec.get("output_length", 0))
        input_sum += iv
        output_sum += ov
        full_input_vals.append(iv)
        full_output_vals.append(ov)

    if total_n == 0:
        raise ValueError("输入文件为空，无法采样")

    target_n = args.sample_size
    if target_n <= 0:
        target_n = max(int(total_n * args.sample_ratio), args.min_sample_size)
    target_n = min(target_n, total_n)

    input_qs = [float(x) for x in args.input_cut_quantiles.split(",") if x.strip()]
    input_qs.sort()
    full_input_vals.sort()
    input_cut_points = [quantile(full_input_vals, q) for q in input_qs]

    # Pass 2: 统计分层计数
    stratum_counts: Dict[Stratum, int] = defaultdict(int)
    for rec in iter_jsonl(args.input):
        ts = float(rec.get("timestamp", 0.0))
        turn = int(rec.get("turn", 0))
        tp = str(rec.get("type", "unknown"))
        iv = int(rec.get("input_length", 0))
        tb = make_time_bin(ts, min_ts, max_ts, args.time_bins)
        ib = make_value_bin(iv, input_cut_points)
        key: Stratum = (turn, tb, tp, ib)
        stratum_counts[key] += 1
        full_time_bin[tb] += 1

    targets = allocate_targets(stratum_counts, target_n, total_n)
    actual_target = sum(targets.values())

    # Pass 3: 分层 reservoir sampling
    seen_by_stratum: Dict[Stratum, int] = defaultdict(int)
    sample_by_stratum: Dict[Stratum, List[Record]] = defaultdict(list)

    for rec in iter_jsonl(args.input):
        ts = float(rec.get("timestamp", 0.0))
        turn = int(rec.get("turn", 0))
        tp = str(rec.get("type", "unknown"))
        iv = int(rec.get("input_length", 0))
        tb = make_time_bin(ts, min_ts, max_ts, args.time_bins)
        ib = make_value_bin(iv, input_cut_points)
        key: Stratum = (turn, tb, tp, ib)

        k = targets.get(key, 0)
        if k <= 0:
            continue

        seen_by_stratum[key] += 1
        seen = seen_by_stratum[key]
        bucket = sample_by_stratum[key]
        if len(bucket) < k:
            bucket.append(rec)
        else:
            j = random.randint(1, seen)
            if j <= k:
                bucket[j - 1] = rec

    sampled_records: List[Record] = []
    for key in sample_by_stratum:
        sampled_records.extend(sample_by_stratum[key])

    sampled_records.sort(
        key=lambda r: (
            float(r.get("timestamp", 0.0)),
            int(r.get("turn", 0)),
            int(r.get("chat_id", -1)),
        )
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for rec in sampled_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 采样集统计
    sample_n = len(sampled_records)
    sample_turn = Counter()
    sample_type = Counter()
    sample_time_bin = Counter()
    sample_input_vals: List[int] = []
    sample_output_vals: List[int] = []
    sample_input_sum = 0
    sample_output_sum = 0
    for rec in sampled_records:
        ts = float(rec.get("timestamp", 0.0))
        turn = int(rec.get("turn", 0))
        tp = str(rec.get("type", "unknown"))
        tb = make_time_bin(ts, min_ts, max_ts, args.time_bins)
        sample_turn[turn] += 1
        sample_type[tp] += 1
        sample_time_bin[tb] += 1
        iv = int(rec.get("input_length", 0))
        ov = int(rec.get("output_length", 0))
        sample_input_sum += iv
        sample_output_sum += ov
        sample_input_vals.append(iv)
        sample_output_vals.append(ov)

    full_output_vals.sort()
    sample_input_vals.sort()
    sample_output_vals.sort()

    full_turn_dist = distribution(full_turn, total_n)
    sample_turn_dist = distribution(sample_turn, sample_n)
    full_type_dist = distribution(full_type, total_n)
    sample_type_dist = distribution(sample_type, sample_n)
    full_time_dist = distribution(full_time_bin, total_n)
    sample_time_dist = distribution(sample_time_bin, sample_n)

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# qwen_traceA_blksz_16 轻量化采样报告\n\n")
        f.write("## 采样配置\n")
        f.write(f"- input: `{args.input}`\n")
        f.write(f"- output: `{args.output}`\n")
        f.write(f"- seed: `{args.seed}`\n")
        f.write(
            f"- stratification: `turn + time_bin({args.time_bins}) + type + input_bin({input_qs})`\n"
        )
        f.write(f"- input_bin_cut_points: `{input_cut_points}`\n")
        f.write(f"- expected_sample_size: `{target_n}`\n")
        f.write(f"- allocated_sample_size: `{actual_target}`\n")
        f.write(f"- actual_sample_size: `{sample_n}`\n\n")

        f.write("## 总体规模\n")
        f.write(f"- full_records: `{total_n}`\n")
        f.write(f"- sample_records: `{sample_n}`\n")
        f.write(f"- sampling_ratio: `{(sample_n / total_n):.4f}`\n")
        f.write(f"- timestamp_range: `{min_ts:.3f} -> {max_ts:.3f}`\n\n")

        f.write("## 分布距离 (L1)\n")
        f.write(f"- turn_distribution_l1: `{l1_distance(full_turn_dist, sample_turn_dist):.6f}`\n")
        f.write(f"- type_distribution_l1: `{l1_distance(full_type_dist, sample_type_dist):.6f}`\n")
        f.write(f"- time_bin_distribution_l1: `{l1_distance(full_time_dist, sample_time_dist):.6f}`\n\n")

        f.write("## 关键统计量\n")
        f.write(
            f"- input_length_mean(full/sample): `{input_sum / total_n:.2f}` / `{sample_input_sum / max(sample_n, 1):.2f}`\n"
        )
        f.write(
            f"- output_length_mean(full/sample): `{output_sum / total_n:.2f}` / `{sample_output_sum / max(sample_n, 1):.2f}`\n"
        )
        f.write(
            f"- input_length_p50/p90/p95(full): `{quantile(full_input_vals, 0.5)}` / `{quantile(full_input_vals, 0.9)}` / `{quantile(full_input_vals, 0.95)}`\n"
        )
        f.write(
            f"- input_length_p50/p90/p95(sample): `{quantile(sample_input_vals, 0.5)}` / `{quantile(sample_input_vals, 0.9)}` / `{quantile(sample_input_vals, 0.95)}`\n"
        )
        f.write(
            f"- output_length_p50/p90/p95(full): `{quantile(full_output_vals, 0.5)}` / `{quantile(full_output_vals, 0.9)}` / `{quantile(full_output_vals, 0.95)}`\n"
        )
        f.write(
            f"- output_length_p50/p90/p95(sample): `{quantile(sample_output_vals, 0.5)}` / `{quantile(sample_output_vals, 0.9)}` / `{quantile(sample_output_vals, 0.95)}`\n\n"
        )

        f.write("## top turns 比例对比\n")
        top_turns = [k for k, _ in full_turn.most_common(15)]
        for t in top_turns:
            f.write(
                f"- turn={t}: full `{full_turn_dist.get(str(t), 0.0):.4f}` | sample `{sample_turn_dist.get(str(t), 0.0):.4f}`\n"
            )
        f.write("\n## type 比例对比\n")
        for tp in sorted(set(full_type.keys()) | set(sample_type.keys())):
            f.write(
                f"- type={tp}: full `{full_type_dist.get(str(tp), 0.0):.4f}` | sample `{sample_type_dist.get(str(tp), 0.0):.4f}`\n"
            )
        f.write("\n## time_bin 比例对比\n")
        for tb in range(args.time_bins):
            f.write(
                f"- bin={tb}: full `{full_time_dist.get(str(tb), 0.0):.4f}` | sample `{sample_time_dist.get(str(tb), 0.0):.4f}`\n"
            )

    print(f"full_records={total_n}")
    print(f"sample_records={sample_n}")
    print(f"sample_ratio={sample_n / total_n:.4f}")
    print(f"output={args.output}")
    print(f"report={args.report}")
    print(f"turn_l1={l1_distance(full_turn_dist, sample_turn_dist):.6f}")
    print(f"type_l1={l1_distance(full_type_dist, sample_type_dist):.6f}")
    print(f"time_l1={l1_distance(full_time_dist, sample_time_dist):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
