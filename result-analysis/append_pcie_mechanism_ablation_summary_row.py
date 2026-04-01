#!/usr/bin/env python3
"""
将一次机制消融实验的汇总指标追加到 TSV 文本表（每个实际跑的 group 一行）。

与 auto_run_mechanism_ablation.sh 产出的 prefetch_<suffix>.jsonl /
pcie_events_<suffix>.json 一一对应；列与 pcie_ablation_experiments 总表一致，便于 Excel。

--groups 须与当次实验的 RUN_GROUPS 一致（逗号分隔，如 no-pq,no-ef 或 all 展开后的列表）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

# (group_id, jsonl/pcie suffix, description) — 与 auto_run_mechanism_ablation.sh 中 run_phase 的 SUFFIX 一致
MECH_GROUP_DEFS: list[tuple[str, str, str]] = [
    ("full", "full_sched", "Complete scheduler (heap, CC=2, evict-first)"),
    ("no-pq", "no_pq", "No priority queue (FIFO)"),
    ("no-ef", "no_ef", "No evict-first"),
    ("no-cc", "no_cc", "No concurrency control (CC=999)"),
    ("g1", "g1_prefetch_only", "Prefetch only, no scheduler"),
    ("g0", "g0_no_prefetch", "Baseline vLLM, no prefetch"),
]

TSV_COLUMNS = [
    "exp_id",
    "model",
    "dataset",
    "qps",
    "lead_time",
    "gpu_blocks",
    "num_conv",
    "gpu_mem_util",
    "pp_size",
    "max_num_seqs",
    "group",
    "group_desc",
    "num_requests",
    "TTFT_mean",
    "TTFT_p50",
    "TTFT_p95",
    "TTFT_p99",
    "TTFT_std",
    "TPOT_mean",
    "TPOT_p50",
    "TPOT_p95",
    "TPOT_p99",
    "h2d_count",
    "h2d_total_gb",
    "h2d_mean_dur_ms",
    "h2d_agg_bw_gbps",
    "h2d_mean_bw_gbps",
    "d2h_count",
    "d2h_total_gb",
    "d2h_mean_dur_ms",
    "p2p_count",
    "pcie_duration_s",
    "prefix_cache_hit_pct",
]

TSV_HEADER = "\t".join(TSV_COLUMNS)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_model_bucket(model_path: str) -> str:
    low = model_path.lower().replace("\\", "/")
    checks = [
        (r"(?<![0-9.])72b(?![0-9])", "72B"),
        (r"(?<![0-9.])32b(?![0-9])", "32B"),
        (r"(?<![0-9.])14b(?![0-9])", "14B"),
        (r"(?<![0-9.])8b(?![0-9])", "8B"),
    ]
    for pat, lab in checks:
        if re.search(pat, low):
            return lab
    return "unknown"


def _percentile_linear(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_vals[0])
    idx = (n - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    w = idx - lo
    return float(sorted_vals[lo] + w * (sorted_vals[hi] - sorted_vals[lo]))


def _float_series(rows: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        if not r.get("success", True):
            continue
        v = r.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        out.append(fv)
    return out


def stat_str(vals: list[float], p: float) -> str:
    if not vals:
        return ""
    return f"{_percentile_linear(sorted(vals), p):.4f}"


def stat_mean(vals: list[float]) -> str:
    if not vals:
        return ""
    return f"{sum(vals) / len(vals):.4f}"


def stat_std(vals: list[float]) -> str:
    if not vals:
        return ""
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    return f"{math.sqrt(var):.4f}"


def analyze_pcie_events(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fp:
        events = json.load(fp)
    if not events:
        return {}

    h2d = [e for e in events if e.get("direction") == "H2D"]
    d2h = [e for e in events if e.get("direction") == "D2H"]
    p2p = [e for e in events if e.get("direction") == "P2P"]

    h2d_bytes = sum(e.get("size_bytes", 0) for e in h2d)
    d2h_bytes = sum(e.get("size_bytes", 0) for e in d2h)
    h2d_gb = h2d_bytes / (1024**3)
    d2h_gb = d2h_bytes / (1024**3)

    h2d_dur_ms = [e.get("duration_ms", 0) for e in h2d if e.get("duration_ms", 0) > 0]
    d2h_dur_ms = [e.get("duration_ms", 0) for e in d2h if e.get("duration_ms", 0) > 0]

    start_us = min((e.get("start_us", 0) for e in events), default=0)
    end_us = max(
        (e.get("start_us", 0) + e.get("duration_ms", 0) * 1000 for e in events),
        default=0,
    )
    duration_s = (end_us - start_us) / 1_000_000 if end_us > start_us else 0.0

    h2d_bw_per = []
    for e in h2d:
        dur_s = e.get("duration_ms", 0) / 1000.0
        if dur_s > 0:
            h2d_bw_per.append(e.get("size_bytes", 0) / (1024**3) / dur_s)

    return {
        "h2d_count": len(h2d),
        "h2d_total_gb": round(h2d_gb, 6),
        "h2d_mean_dur_ms": round(sum(h2d_dur_ms) / len(h2d_dur_ms), 4) if h2d_dur_ms else 0.0,
        "d2h_count": len(d2h),
        "d2h_total_gb": round(d2h_gb, 6),
        "d2h_mean_dur_ms": round(sum(d2h_dur_ms) / len(d2h_dur_ms), 4) if d2h_dur_ms else 0.0,
        "p2p_count": len(p2p),
        "duration_s": round(duration_s, 4),
        "h2d_agg_bw_gbps": round(h2d_gb / duration_s, 4) if duration_s > 0 else 0.0,
        "h2d_mean_bw_gbps": round(sum(h2d_bw_per) / len(h2d_bw_per), 4) if h2d_bw_per else 0.0,
    }


def prefix_cache_hit_rate_pct(rows: list[dict]) -> str:
    if not rows:
        return ""
    hit = 0
    for r in rows:
        if r.get("cached_tokens", 0) or r.get("prefetch_cached_tokens", 0):
            hit += 1
    return f"{hit / len(rows) * 100:.4f}"


def tsv_cell(s: str) -> str:
    return (s or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")


def build_row(
    args: argparse.Namespace,
    model: str,
    group_name: str,
    group_desc: str,
    rows: list[dict],
    pcie: dict[str, Any],
) -> str:
    ttft = _float_series(rows, "ttft_ms")
    tpot = _float_series(rows, "tpot_ms")

    def pv(key: str) -> str:
        v = pcie.get(key)
        return str(v) if v is not None else ""

    cells = [
        args.exp_id,
        model,
        args.dataset,
        args.qps,
        args.lead_time,
        args.num_gpu_blocks,
        args.num_conv,
        args.gpu_mem_util,
        args.vllm_pp,
        args.vllm_max_num_seqs,
        group_name,
        group_desc,
        str(len(rows)),
        stat_mean(ttft),
        stat_str(ttft, 50),
        stat_str(ttft, 95),
        stat_str(ttft, 99),
        stat_std(ttft),
        stat_mean(tpot),
        stat_str(tpot, 50),
        stat_str(tpot, 95),
        stat_str(tpot, 99),
        pv("h2d_count"),
        pv("h2d_total_gb"),
        pv("h2d_mean_dur_ms"),
        pv("h2d_agg_bw_gbps"),
        pv("h2d_mean_bw_gbps"),
        pv("d2h_count"),
        pv("d2h_total_gb"),
        pv("d2h_mean_dur_ms"),
        pv("p2p_count"),
        pv("duration_s"),
        prefix_cache_hit_rate_pct(rows),
    ]
    return "\t".join(tsv_cell(str(c)) for c in cells)


def resolve_groups(groups_csv: str) -> list[tuple[str, str, str]]:
    requested = {x.strip().lower() for x in groups_csv.split(",") if x.strip()}
    if not requested:
        raise SystemExit("empty --groups")
    out: list[tuple[str, str, str]] = []
    for gid, suffix, desc in MECH_GROUP_DEFS:
        if gid.lower() in requested:
            out.append((gid, suffix, desc))
    unknown = requested - {g[0].lower() for g in MECH_GROUP_DEFS}
    if unknown:
        raise SystemExit(f"unknown group id(s) in --groups: {sorted(unknown)}")
    if not out:
        raise SystemExit("no groups matched MECH_GROUP_DEFS")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Append mechanism-ablation rows to a TSV summary table"
    )
    p.add_argument("--results-dir", required=True)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--qps", required=True)
    p.add_argument("--lead-time", required=True)
    p.add_argument("--num-gpu-blocks", required=True)
    p.add_argument("--num-conv", required=True)
    p.add_argument("--gpu-mem-util", required=True)
    p.add_argument("--vllm-pp", required=True)
    p.add_argument("--vllm-max-num-seqs", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--table", required=True, help="TSV output path")
    p.add_argument(
        "--groups",
        required=True,
        help="Comma-separated groups (e.g. no-pq,no-ef or g0,g1,full,...)",
    )
    args = p.parse_args()

    group_rows = resolve_groups(args.groups)
    base = Path(args.results_dir)
    model = infer_model_bucket(args.model_path)
    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not table_path.exists() or table_path.stat().st_size == 0

    lines: list[str] = []
    if write_header:
        lines.append(TSV_HEADER)

    for group_name, suffix, group_desc in group_rows:
        jsonl_path = base / f"prefetch_{suffix}.jsonl"
        pcie_path = base / f"pcie_events_{suffix}.json"
        rows = load_jsonl(jsonl_path)
        pcie = analyze_pcie_events(pcie_path)
        lines.append(build_row(args, model, group_name, group_desc, rows, pcie))

    with open(table_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Appended {len(group_rows)} rows to {table_path}")


if __name__ == "__main__":
    main()
