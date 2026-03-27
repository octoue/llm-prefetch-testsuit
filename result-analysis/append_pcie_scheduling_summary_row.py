#!/usr/bin/env python3
"""
将单次 PCIe 调度 A/B 实验的汇总指标追加到 TSV 文本表（便于粘贴到 Excel）。

列顺序与 run_pcie_scheduling_ab.sh 约定一致；首次写入时自动写入表头。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


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
    """从路径推断 8B / 32B / 72B；避免 1.8B 等被误判为 8B。"""
    low = model_path.lower().replace("\\", "/")
    checks = [
        (r"(?<![0-9.])72b(?![0-9])", "72B"),
        (r"(?<![0-9.])32b(?![0-9])", "32B"),
        (r"(?<![0-9.])8b(?![0-9])", "8B"),
    ]
    for pat, lab in checks:
        if re.search(pat, low):
            return lab
    return "unknown"


def _percentile_linear(sorted_vals: list[float], p: float) -> float:
    """与 numpy.percentile(..., method="linear") 一致的分位数。"""
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


def stat_mean_p50_p95_p99(vals: list[float]) -> tuple[str, str, str, str]:
    if not vals:
        return ("", "", "", "")
    s = sorted(vals)
    mean = sum(s) / len(s)
    return (
        f"{mean:.4f}",
        f"{_percentile_linear(s, 50):.4f}",
        f"{_percentile_linear(s, 95):.4f}",
        f"{_percentile_linear(s, 99):.4f}",
    )


def analyze_pcie_events(events: list[dict]) -> dict[str, Any]:
    if not events:
        return {}
    h2d = [e for e in events if e.get("direction") == "H2D"]
    d2h = [e for e in events if e.get("direction") == "D2H"]
    h2d_gb = sum(e.get("size_bytes", 0) for e in h2d) / (1024**3)
    d2h_gb = sum(e.get("size_bytes", 0) for e in d2h) / (1024**3)
    start_us = min(e.get("start_us", 0) for e in events)
    end_us = max(
        e.get("start_us", 0) + e.get("duration_ms", 0) * 1000 for e in events
    )
    duration_s = (end_us - start_us) / 1_000_000 if end_us > start_us else 0.0
    return {
        "h2d_count": len(h2d),
        "h2d_total_gb": round(h2d_gb, 6),
        "d2h_count": len(d2h),
        "d2h_total_gb": round(d2h_gb, 6),
        "duration_s": round(duration_s, 6),
        "h2d_bandwidth_gbps": round(h2d_gb * 8 / duration_s, 6) if duration_s > 0 else 0.0,
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


# 表头（与下方 row 顺序严格一致）
TSV_HEADER = "\t".join(
    [
        "实验辨识码",
        "模型",
        "dataset",
        "qps",
        "PREFETCH_LEAD_TIME",
        "NUM_GPU_BLOCKS_OVERRIDE",
        "num_conv",
        "GPU_MEMORY_UTILIZATION",
        "VLLM_PIPELINE_PARALLEL_SIZE",
        "VLLM_MAX_NUM_SEQS",
        "TTFT_mean_PCIeSched",
        "TTFT_mean_Baseline",
        "TTFT_p50_PCIeSched",
        "TTFT_p50_Baseline",
        "TTFT_p95_PCIeSched",
        "TTFT_p95_Baseline",
        "TTFT_p99_PCIeSched",
        "TTFT_p99_Baseline",
        "TPOT_mean_PCIeSched",
        "TPOT_mean_Baseline",
        "TPOT_p50_PCIeSched",
        "TPOT_p50_Baseline",
        "TPOT_p95_PCIeSched",
        "TPOT_p95_Baseline",
        "TPOT_p99_PCIeSched",
        "TPOT_p99_Baseline",
        "PCIe_h2d_count_PCIeSched",
        "PCIe_h2d_count_Baseline",
        "PCIe_h2d_total_gb_PCIeSched",
        "PCIe_h2d_total_gb_Baseline",
        "PCIe_d2h_count_PCIeSched",
        "PCIe_d2h_count_Baseline",
        "PCIe_d2h_total_gb_PCIeSched",
        "PCIe_d2h_total_gb_Baseline",
        "PCIe_h2d_bandwidth_gbps_PCIeSched",
        "PCIe_h2d_bandwidth_gbps_Baseline",
        "PCIe_duration_s_PCIeSched",
        "PCIe_duration_s_Baseline",
        "Prefix_cache_hit_rate_pct_PCIeSched",
        "Prefix_cache_hit_rate_pct_Baseline",
    ]
)


def main() -> None:
    p = argparse.ArgumentParser()
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
    p.add_argument("--table", required=True, help="TSV 汇总表路径（.txt）")
    args = p.parse_args()

    base = Path(args.results_dir)
    sched_rows = load_jsonl(base / "prefetch_pcie_sched.jsonl")
    base_rows = load_jsonl(base / "prefetch_baseline.jsonl")

    sm_s, s50_s, s95_s, s99_s = stat_mean_p50_p95_p99(_float_series(sched_rows, "ttft_ms"))
    sm_b, s50_b, s95_b, s99_b = stat_mean_p50_p95_p99(_float_series(base_rows, "ttft_ms"))

    tpm_s, tp50_s, tp95_s, tp99_s = stat_mean_p50_p95_p99(
        _float_series(sched_rows, "tpot_ms")
    )
    tpm_b, tp50_b, tp95_b, tp99_b = stat_mean_p50_p95_p99(
        _float_series(base_rows, "tpot_ms")
    )

    pcie_sched: list[dict] = []
    pcie_base: list[dict] = []
    f_sched = base / "pcie_events_pcie_sched.json"
    f_base = base / "pcie_events_baseline.json"
    if f_sched.exists():
        with open(f_sched, encoding="utf-8") as fp:
            pcie_sched = json.load(fp)
    if f_base.exists():
        with open(f_base, encoding="utf-8") as fp:
            pcie_base = json.load(fp)

    st_s = analyze_pcie_events(pcie_sched)
    st_b = analyze_pcie_events(pcie_base)

    def fmt_pcie_val(st: dict[str, Any], key: str) -> str:
        if not st or key not in st:
            return ""
        return str(st[key])

    h2dc_s, h2dc_b = fmt_pcie_val(st_s, "h2d_count"), fmt_pcie_val(st_b, "h2d_count")
    h2dg_s, h2dg_b = fmt_pcie_val(st_s, "h2d_total_gb"), fmt_pcie_val(st_b, "h2d_total_gb")
    d2hc_s, d2hc_b = fmt_pcie_val(st_s, "d2h_count"), fmt_pcie_val(st_b, "d2h_count")
    d2hg_s, d2hg_b = fmt_pcie_val(st_s, "d2h_total_gb"), fmt_pcie_val(st_b, "d2h_total_gb")
    bw_s, bw_b = fmt_pcie_val(st_s, "h2d_bandwidth_gbps"), fmt_pcie_val(
        st_b, "h2d_bandwidth_gbps"
    )
    dur_s, dur_b = fmt_pcie_val(st_s, "duration_s"), fmt_pcie_val(st_b, "duration_s")

    hit_s = prefix_cache_hit_rate_pct(sched_rows)
    hit_b = prefix_cache_hit_rate_pct(base_rows)
    model = infer_model_bucket(args.model_path)

    row_cells = [
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
        sm_s,
        sm_b,
        s50_s,
        s50_b,
        s95_s,
        s95_b,
        s99_s,
        s99_b,
        tpm_s,
        tpm_b,
        tp50_s,
        tp50_b,
        tp95_s,
        tp95_b,
        tp99_s,
        tp99_b,
        h2dc_s,
        h2dc_b,
        h2dg_s,
        h2dg_b,
        d2hc_s,
        d2hc_b,
        d2hg_s,
        d2hg_b,
        bw_s,
        bw_b,
        dur_s,
        dur_b,
        hit_s,
        hit_b,
    ]
    line = "\t".join(tsv_cell(str(c)) for c in row_cells)

    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not table_path.exists() or table_path.stat().st_size == 0
    with open(table_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(TSV_HEADER + "\n")
        f.write(line + "\n")

    print(f"Appended row to {table_path}")


if __name__ == "__main__":
    main()
