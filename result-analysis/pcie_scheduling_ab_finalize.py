#!/usr/bin/env python3
"""
PCIe 调度 A/B 跑批收尾：生成合并 Markdown 报告，并向汇总 TSV（制表符分隔，可粘贴 Excel）追加一行。

不依赖 config_snapshot.env；配置由命令行传入。仅使用标准库 + 同目录 generate_pcie_scheduling_report。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_RAP = Path(__file__).resolve().parent
if str(_RAP) not in sys.path:
    sys.path.insert(0, str(_RAP))

from generate_pcie_scheduling_report import (
    analyze_pcie_events,
    extract_pcie_scheduler_stats,
    load_jsonl,
)


def infer_model_size(model_path: str) -> str:
    p = model_path.lower()
    for tag, name in (("72b", "72B"), ("32b", "32B"), ("8b", "8B")):
        if tag in p:
            return name
    return "unknown"


def _success_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("success", True):
            out.append(r)
    return out


def _latency_stats(rows: list[dict], key: str) -> dict[str, float]:
    vals = sorted(
        float(r[key])
        for r in rows
        if key in r and r[key] is not None
    )
    n = len(vals)
    if n == 0:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "mean": sum(vals) / n,
        "p50": vals[n // 2],
        "p95": vals[int(n * 0.95)] if n else float("nan"),
        "p99": vals[int(n * 0.99)] if n else float("nan"),
    }


def _prefix_hit_pct(rows: list[dict]) -> float:
    sub = [r for r in rows if r.get("cached_tokens") is not None]
    if not sub:
        return float("nan")
    hits = sum(1 for r in sub if (r.get("cached_tokens") or 0) > 0)
    return hits / len(sub) * 100


def _format_config_md(
    experiment_id: str,
    results_dir: Path,
    kwargs: dict[str, str],
) -> list[str]:
    order = [
        ("experiment_id", "实验辨识码"),
        ("results_dir", "结果目录"),
        ("model_size", "模型规模"),
        ("model_path", "MODEL_PATH"),
        ("dataset", "DATASET"),
        ("qps", "QPS"),
        ("prefetch_lead_time", "PREFETCH_LEAD_TIME"),
        ("num_gpu_blocks_override", "NUM_GPU_BLOCKS_OVERRIDE"),
        ("num_conv", "NUM_CONV"),
        ("gpu_memory_utilization", "GPU_MEMORY_UTILIZATION"),
        ("vllm_pipeline_parallel_size", "VLLM_PIPELINE_PARALLEL_SIZE"),
        ("vllm_max_num_seqs", "VLLM_MAX_NUM_SEQS"),
        ("pp_phase_h2d_policy", "PP_PHASE_H2D_POLICY"),
    ]
    lines = [
        "## 1. 配置与实验标识",
        "",
        "| 参数 | 值 |",
        "|------|-----|",
    ]
    for key, label in order:
        if key == "results_dir":
            v = str(results_dir)
        else:
            v = kwargs.get(key, "")
        lines.append(f"| {label} | {v} |")
    lines.append("")
    return lines


def _ttft_tpot_tsv_and_md(
    plain: list[dict],
    baseline: list[dict],
    pcie: list[dict],
    three_way: bool,
) -> tuple[dict[str, Any], list[str]]:
    """TSV 始终含 plain / prefetch / pcie_sched 三列，便于同一汇总表多次追加。"""
    tsv: dict[str, Any] = {}

    def put_phase(prefix: str, rows: list[dict]) -> None:
        tt = _latency_stats(rows, "ttft_ms")
        tp = _latency_stats(rows, "tpot_ms")
        tsv[f"ttft_mean_{prefix}"] = tt["mean"]
        tsv[f"ttft_p50_{prefix}"] = tt["p50"]
        tsv[f"ttft_p95_{prefix}"] = tt["p95"]
        tsv[f"ttft_p99_{prefix}"] = tt["p99"]
        tsv[f"tpot_mean_{prefix}"] = tp["mean"]
        tsv[f"tpot_p50_{prefix}"] = tp["p50"]
        tsv[f"tpot_p95_{prefix}"] = tp["p95"]
        tsv[f"tpot_p99_{prefix}"] = tp["p99"]
        tsv[f"prefix_cache_hit_pct_{prefix}"] = _prefix_hit_pct(rows)

    put_phase("plain", plain)
    put_phase("prefetch", baseline)
    put_phase("pcie_sched", pcie)

    def fmt_row_ttft(rows: list[dict]) -> dict[str, float]:
        return _latency_stats(rows, "ttft_ms")

    def fmt_row_tpot(rows: list[dict]) -> dict[str, float]:
        return _latency_stats(rows, "tpot_ms")

    lines: list[str] = []
    if three_way:
        tp = fmt_row_ttft(plain)
        tb = fmt_row_ttft(baseline)
        ts = fmt_row_ttft(pcie)
        lines.extend(
            [
                "## 2. TTFT（ms）",
                "",
                "| 统计 | 普通 vLLM | +Prefetch | +Prefetch+PCIe |",
                "|------|-----------|-----------|----------------|",
                f"| Mean | {_fmt_ms(tp['mean'])} | {_fmt_ms(tb['mean'])} | {_fmt_ms(ts['mean'])} |",
                f"| P50 | {_fmt_ms(tp['p50'])} | {_fmt_ms(tb['p50'])} | {_fmt_ms(ts['p50'])} |",
                f"| P95 | {_fmt_ms(tp['p95'])} | {_fmt_ms(tb['p95'])} | {_fmt_ms(ts['p95'])} |",
                f"| P99 | {_fmt_ms(tp['p99'])} | {_fmt_ms(tb['p99'])} | {_fmt_ms(ts['p99'])} |",
                "",
            ]
        )
        pp = fmt_row_tpot(plain)
        pb = fmt_row_tpot(baseline)
        ps = fmt_row_tpot(pcie)
        lines.extend(
            [
                "## 3. TPOT（ms）",
                "",
                "| 统计 | 普通 vLLM | +Prefetch | +Prefetch+PCIe |",
                "|------|-----------|-----------|----------------|",
                f"| Mean | {_fmt_ms(pp['mean'])} | {_fmt_ms(pb['mean'])} | {_fmt_ms(ps['mean'])} |",
                f"| P50 | {_fmt_ms(pp['p50'])} | {_fmt_ms(pb['p50'])} | {_fmt_ms(ps['p50'])} |",
                f"| P95 | {_fmt_ms(pp['p95'])} | {_fmt_ms(pb['p95'])} | {_fmt_ms(ps['p95'])} |",
                f"| P99 | {_fmt_ms(pp['p99'])} | {_fmt_ms(pb['p99'])} | {_fmt_ms(ps['p99'])} |",
                "",
                "## 4. Prefix cache（cached_tokens>0 请求占比 %）",
                "",
                "| 普通 vLLM | +Prefetch | +Prefetch+PCIe |",
                "|-----------|-----------|----------------|",
                f"| {_fmt_pct(_prefix_hit_pct(plain))} | {_fmt_pct(_prefix_hit_pct(baseline))} | {_fmt_pct(_prefix_hit_pct(pcie))} |",
                "",
            ]
        )
    else:
        tb = fmt_row_ttft(baseline)
        ts = fmt_row_ttft(pcie)
        lines.extend(
            [
                "## 2. TTFT（ms）",
                "",
                "| 统计 | +Prefetch | +Prefetch+PCIe |",
                "|------|-----------|----------------|",
                f"| Mean | {_fmt_ms(tb['mean'])} | {_fmt_ms(ts['mean'])} |",
                f"| P50 | {_fmt_ms(tb['p50'])} | {_fmt_ms(ts['p50'])} |",
                f"| P95 | {_fmt_ms(tb['p95'])} | {_fmt_ms(ts['p95'])} |",
                f"| P99 | {_fmt_ms(tb['p99'])} | {_fmt_ms(ts['p99'])} |",
                "",
            ]
        )
        pb = fmt_row_tpot(baseline)
        ps = fmt_row_tpot(pcie)
        lines.extend(
            [
                "## 3. TPOT（ms）",
                "",
                "| 统计 | +Prefetch | +Prefetch+PCIe |",
                "|------|-----------|----------------|",
                f"| Mean | {_fmt_ms(pb['mean'])} | {_fmt_ms(ps['mean'])} |",
                f"| P50 | {_fmt_ms(pb['p50'])} | {_fmt_ms(ps['p50'])} |",
                f"| P95 | {_fmt_ms(pb['p95'])} | {_fmt_ms(ps['p95'])} |",
                f"| P99 | {_fmt_ms(pb['p99'])} | {_fmt_ms(ps['p99'])} |",
                "",
                "## 4. Prefix cache（cached_tokens>0 请求占比 %）",
                "",
                "| +Prefetch | +Prefetch+PCIe |",
                "|-----------|----------------|",
                f"| {_fmt_pct(_prefix_hit_pct(baseline))} | {_fmt_pct(_prefix_hit_pct(pcie))} |",
                "",
            ]
        )

    return tsv, lines


def _fmt_ms(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.2f}"


def _fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.2f}"


def _pcie_tsv_and_md(
    base: Path, three_way: bool
) -> tuple[dict[str, Any], list[str]]:
    keys = [
        "h2d_count",
        "h2d_total_gb",
        "d2h_count",
        "d2h_total_gb",
        "h2d_bandwidth_gbps",
        "d2h_bandwidth_gbps",
        "duration_s",
    ]
    pcie_events_sched: list[dict] = []
    pcie_events_base: list[dict] = []
    pcie_events_plain: list[dict] = []
    if (base / "pcie_events_pcie_sched.json").exists():
        with open(base / "pcie_events_pcie_sched.json", encoding="utf-8") as f:
            pcie_events_sched = json.load(f)
    if (base / "pcie_events_baseline.json").exists():
        with open(base / "pcie_events_baseline.json", encoding="utf-8") as f:
            pcie_events_base = json.load(f)
    if (base / "pcie_events_plain.json").exists():
        with open(base / "pcie_events_plain.json", encoding="utf-8") as f:
            pcie_events_plain = json.load(f)

    st_sched = analyze_pcie_events(pcie_events_sched)
    st_base = analyze_pcie_events(pcie_events_base)
    st_plain = analyze_pcie_events(pcie_events_plain)

    tsv: dict[str, Any] = {}
    for k in keys:
        for pref, st in (
            ("plain", st_plain),
            ("prefetch", st_base),
            ("pcie_sched", st_sched),
        ):
            tsv[f"pcie_{k}_{pref}"] = st.get(k, "") if st else ""

    lines = [
        "## 5. PCIe 事件汇总（profiler）",
        "",
    ]
    if three_way:
        lines.extend(
            [
                "| 指标 | 普通 vLLM | +Prefetch | +Prefetch+PCIe |",
                "|------|-----------|-----------|----------------|",
            ]
        )
        for k in keys:
            a = st_plain.get(k, "-")
            b = st_base.get(k, "-")
            c = st_sched.get(k, "-")
            lines.append(
                f"| {k} | {_fmt_pcie_val(a)} | {_fmt_pcie_val(b)} | {_fmt_pcie_val(c)} |"
            )
    else:
        lines.extend(
            [
                "| 指标 | +Prefetch | +Prefetch+PCIe |",
                "|------|-----------|----------------|",
            ]
        )
        for k in keys:
            b = st_base.get(k, "-")
            c = st_sched.get(k, "-")
            lines.append(
                f"| {k} | {_fmt_pcie_val(b)} | {_fmt_pcie_val(c)} |"
            )
    lines.append("")
    return tsv, lines


def _fmt_pcie_val(v: Any) -> str:
    if v == "-" or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _scheduler_md(base: Path) -> list[str]:
    sched = extract_pcie_scheduler_stats(base / "vllm_state_pcie_sched.log")
    lines = [
        "## 6. PCIe Scheduler 日志统计（+Prefetch+PCIe 阶段，若有）",
        "",
    ]
    if not sched:
        lines.append("*未在 vllm 日志中找到 PCIe Scheduler Stats*")
        lines.append("")
        return lines
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    for k, v in sorted(sched.items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return lines


def _paired_and_turn_md(baseline: list[dict], pcie: list[dict]) -> list[str]:
    def key(r: dict) -> tuple[Any, Any]:
        return (r.get("chat_id"), r.get("turn"))

    by_b = {key(r): r for r in baseline if "ttft_ms" in r}
    merged: list[tuple[float, float]] = []
    for r in pcie:
        k = key(r)
        if k in by_b and "ttft_ms" in r:
            merged.append((float(by_b[k]["ttft_ms"]), float(r["ttft_ms"])))

    mean_diff = float("nan")
    if merged:
        mean_diff = sum(a - b for a, b in merged) / len(merged)

    by_turn: dict[Any, list[tuple[float, float]]] = {}
    for r in pcie:
        k = key(r)
        if k not in by_b or "ttft_ms" not in r:
            continue
        t = r.get("turn")
        by_turn.setdefault(t, []).append(
            (float(by_b[k]["ttft_ms"]), float(r["ttft_ms"]))
        )

    turn_lines = []
    for turn_val in sorted(by_turn.keys(), key=lambda x: (x is None, x)):
        sub = by_turn[turn_val]
        tb = sum(x[0] for x in sub) / len(sub)
        tp = sum(x[1] for x in sub) / len(sub)
        turn_lines.append(f"| {turn_val} | {tb:.2f} | {tp:.2f} |")
    turn_table = (
        "\n".join(turn_lines)
        if turn_lines
        else "| (无配对样本) | - | - |"
    )
    return [
        "## 7. 配对对比（+Prefetch vs +Prefetch+PCIe）",
        "",
        f"- 配对样本数: {len(merged)}",
        f"- 平均 TTFT 差值 (+Prefetch − +Prefetch+PCIe): {_fmt_ms(mean_diff)} ms",
        "",
        "## 8. 逐 Turn TTFT（配对）",
        "",
        "| Turn | +Prefetch (ms) | +Prefetch+PCIe (ms) |",
        "|------|----------------|---------------------|",
        turn_table,
        "",
    ]


def build_tsv_header() -> list[str]:
    base_cols = [
        "experiment_id",
        "model_size",
        "dataset",
        "qps",
        "prefetch_lead_time",
        "num_gpu_blocks_override",
        "num_conv",
        "gpu_memory_utilization",
        "vllm_pipeline_parallel_size",
        "vllm_max_num_seqs",
        "pp_phase_h2d_policy",
    ]
    phases = ["plain", "prefetch", "pcie_sched"]

    def triple(prefix: str, stats: list[str]) -> list[str]:
        out = []
        for s in stats:
            for ph in phases:
                out.append(f"{prefix}_{s}_{ph}")
        return out

    ttft_stats = ["mean", "p50", "p95", "p99"]
    tpot_stats = ["mean", "p50", "p95", "p99"]
    pcie_keys = [
        "h2d_count",
        "h2d_total_gb",
        "d2h_count",
        "d2h_total_gb",
        "h2d_bandwidth_gbps",
        "d2h_bandwidth_gbps",
        "duration_s",
    ]
    return (
        base_cols
        + triple("ttft", ttft_stats)
        + triple("tpot", tpot_stats)
        + [f"pcie_{k}_{ph}" for k in pcie_keys for ph in phases]
        + [f"prefix_cache_hit_pct_{ph}" for ph in phases]
    )


def append_tsv_row(tsv_path: Path, header: list[str], row: list[Any]) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
    with open(tsv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(_tsv_cell(x) for x in row) + "\n")


def _tsv_cell(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    if isinstance(x, float):
        if x == int(x):
            return str(int(x))
        return f"{x:.6g}".rstrip("0").rstrip(".")
    s = str(x)
    if "\t" in s or "\n" in s or "\r" in s:
        s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return s


def build_row_dict(
    args: argparse.Namespace,
    model_size: str,
    tsv_ttft: dict[str, Any],
    tsv_pcie: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "model_size": model_size,
        "dataset": args.dataset,
        "qps": args.qps,
        "prefetch_lead_time": args.lead_time,
        "num_gpu_blocks_override": args.num_gpu_blocks,
        "num_conv": args.num_conv,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "vllm_pipeline_parallel_size": args.vllm_pipeline_parallel_size,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "pp_phase_h2d_policy": args.pp_phase_h2d_policy,
    }
    row.update(tsv_ttft)
    row.update(tsv_pcie)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="PCIe A/B 合并报告 + TSV 汇总行")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--qps", required=True)
    parser.add_argument("--lead-time", required=True)
    parser.add_argument("--num-gpu-blocks", required=True)
    parser.add_argument("--num-conv", required=True)
    parser.add_argument("--gpu-memory-utilization", required=True)
    parser.add_argument("--vllm-pipeline-parallel-size", required=True)
    parser.add_argument("--vllm-max-num-seqs", required=True)
    parser.add_argument(
        "--pp-phase-h2d-policy",
        default="soft",
        choices=["soft", "hard", "restore_only"],
        help="与 Phase 3 start_vllm_pcie.sh 使用的 --pp-phase-h2d-policy 一致（vLLM SchedulerConfig）",
    )
    parser.add_argument(
        "--md-output",
        default="",
        help="默认: <results-dir>/experiment_report.md",
    )
    parser.add_argument(
        "--tsv-path",
        required=True,
        help="汇总表路径（制表符分隔），不存在则写表头",
    )
    args = parser.parse_args()

    base = Path(args.results_dir).resolve()
    md_out = Path(args.md_output) if args.md_output else base / "experiment_report.md"
    tsv_path = Path(args.tsv_path).resolve()

    model_size = infer_model_size(args.model_path)

    rows_p = _success_rows(load_jsonl(base / "prefetch_pcie_sched.jsonl"))
    rows_b = _success_rows(load_jsonl(base / "prefetch_baseline.jsonl"))
    rows_plain = _success_rows(load_jsonl(base / "plain_vllm.jsonl"))
    three_way = len(rows_plain) > 0

    cfg_kwargs = {
        "experiment_id": args.experiment_id,
        "model_size": model_size,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "qps": str(args.qps),
        "prefetch_lead_time": str(args.lead_time),
        "num_gpu_blocks_override": str(args.num_gpu_blocks),
        "num_conv": str(args.num_conv),
        "gpu_memory_utilization": str(args.gpu_memory_utilization),
        "vllm_pipeline_parallel_size": str(args.vllm_pipeline_parallel_size),
        "vllm_max_num_seqs": str(args.vllm_max_num_seqs),
        "pp_phase_h2d_policy": str(args.pp_phase_h2d_policy),
    }

    tsv_ttft, md_ttft = _ttft_tpot_tsv_and_md(
        rows_plain, rows_b, rows_p, three_way
    )
    tsv_pcie, md_pcie = _pcie_tsv_and_md(base, three_way)

    title = (
        "# PCIe 调度 A/B 实验报告（普通 vLLM / +Prefetch / +Prefetch+PCIe）"
        if three_way
        else "# PCIe 调度 A/B 实验报告（+Prefetch / +Prefetch+PCIe）"
    )
    n_plain, n_b, n_p = len(rows_plain), len(rows_b), len(rows_p)
    cfg_block = _format_config_md(args.experiment_id, base, cfg_kwargs)
    if three_way:
        extra_rows = [
            "| 成功样本（普通 vLLM） | "
            f"{n_plain} |",
            "| 成功样本（+Prefetch） | "
            f"{n_b} |",
            "| 成功样本（+Prefetch+PCIe） | "
            f"{n_p} |",
            "",
        ]
    else:
        extra_rows = [
            "| 成功样本（+Prefetch） | "
            f"{n_b} |",
            "| 成功样本（+Prefetch+PCIe） | "
            f"{n_p} |",
            "",
        ]
    cfg_block = cfg_block[:-1] + extra_rows

    md_lines = (
        [title, ""]
        + cfg_block
        + md_ttft
        + md_pcie
        + _scheduler_md(base)
        + _paired_and_turn_md(rows_b, rows_p)
    )

    md_out.parent.mkdir(parents=True, exist_ok=True)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    header = build_tsv_header()
    row_dict = build_row_dict(args, model_size, tsv_ttft, tsv_pcie)
    row = [row_dict.get(h, "") for h in header]
    append_tsv_row(tsv_path, header, row)

    print(f"Markdown: {md_out}")
    print(f"TSV 已追加: {tsv_path}")


if __name__ == "__main__":
    main()
