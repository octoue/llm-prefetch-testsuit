#!/usr/bin/env python3
"""
PCIe 调度 A/B 实验报告生成器

读取 prefetch_pcie_sched.jsonl、prefetch_baseline.jsonl 及可选的 PCIe 事件文件，
生成含配置快照、TTFT/TPOT 对比、PCIe 带宽、Prefetch 行为、稳定性等详细报告。

用法:
  python generate_pcie_scheduling_report.py --results-dir results/xxx --dataset pcie-medium --qps 1.0 --lead-time 2.0 --output report.md
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_config(path: Path) -> dict[str, str]:
    config = {}
    if not path.exists():
        return config
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()
    return config


def analyze_pcie_events(events: list[dict]) -> dict[str, Any]:
    if not events:
        return {}
    h2d = [e for e in events if e.get("direction") == "H2D"]
    d2h = [e for e in events if e.get("direction") == "D2H"]
    p2p = [e for e in events if e.get("direction") == "P2P"]
    h2d_gb = sum(e.get("size_bytes", 0) for e in h2d) / (1024**3)
    d2h_gb = sum(e.get("size_bytes", 0) for e in d2h) / (1024**3)
    p2p_gb = sum(e.get("size_bytes", 0) for e in p2p) / (1024**3)
    start_us = min(e.get("start_us", 0) for e in events)
    end_us = max(
        e.get("start_us", 0) + e.get("duration_ms", 0) * 1000 for e in events
    )
    duration_s = (end_us - start_us) / 1_000_000 if end_us > start_us else 0
    return {
        "h2d_count": len(h2d),
        "h2d_total_gb": round(h2d_gb, 3),
        "d2h_count": len(d2h),
        "d2h_total_gb": round(d2h_gb, 3),
        "p2p_count": len(p2p),
        "p2p_total_gb": round(p2p_gb, 3),
        "duration_s": round(duration_s, 2),
        "h2d_bandwidth_gbps": round(h2d_gb * 8 / duration_s, 2) if duration_s > 0 else 0,
        "d2h_bandwidth_gbps": round(d2h_gb * 8 / duration_s, 2) if duration_s > 0 else 0,
    }


def compute_ttft_stats(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    success = [r for r in rows if r.get("success", True) and "ttft_ms" in r]
    if not success:
        return {}
    ttft = [r["ttft_ms"] for r in success]
    ttft.sort()
    n = len(ttft)
    return {
        "count": n,
        "mean": sum(ttft) / n,
        "p50": ttft[n // 2] if n else 0,
        "p90": ttft[int(n * 0.9)] if n else 0,
        "p95": ttft[int(n * 0.95)] if n else 0,
        "p99": ttft[int(n * 0.99)] if n else 0,
        "std": (sum((x - sum(ttft) / n) ** 2 for x in ttft) / n) ** 0.5 if n > 1 else 0,
    }


def compute_tpot_stats(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    with_tpot = [r for r in rows if r.get("success", True) and r.get("tpot_ms")]
    if not with_tpot:
        return {}
    tpot = [r["tpot_ms"] for r in with_tpot]
    tpot.sort()
    n = len(tpot)
    return {
        "count": n,
        "mean": sum(tpot) / n,
        "p50": tpot[n // 2] if n else 0,
    }


def extract_pcie_scheduler_stats(log_path: Path) -> dict[str, int] | None:
    """Extract PCIe Scheduler Stats from vllm log file.

    Looks for lines like:
    INFO ... PCIe Scheduler Stats: submitted=300, deferred=12, restore=80, prefetch=140, evict=80, max_queue=5, throttled=15, pp_idle_flushes=0, prefetch_starved=0

    Returns the last occurrence (most recent stats).
    """
    if not log_path.exists():
        return None

    last_stats = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "PCIe Scheduler Stats:" in line:
                # Parse the stats from the log line
                stats_part = line.split("PCIe Scheduler Stats:")[1].strip()
                stats = {}
                for pair in stats_part.split(","):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        try:
                            stats[k.strip()] = int(v.strip())
                        except ValueError:
                            pass
                if stats:
                    last_stats = stats
    return last_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PCIe Scheduling A/B Report"
    )
    parser.add_argument("--results-dir", required=True, help="Results directory")
    parser.add_argument("--dataset", default="", help="Dataset name")
    parser.add_argument("--qps", default="", help="QPS value")
    parser.add_argument("--lead-time", default="", help="Prefetch lead time")
    parser.add_argument("--output", required=True, help="Output markdown path")
    args = parser.parse_args()

    base = Path(args.results_dir)
    pcie_sched = load_jsonl(base / "prefetch_pcie_sched.jsonl")
    baseline = load_jsonl(base / "prefetch_baseline.jsonl")
    config = load_config(base / "config_snapshot.env")

    ttft_pcie = compute_ttft_stats(pcie_sched)
    ttft_base = compute_ttft_stats(baseline)
    tpot_pcie = compute_tpot_stats(pcie_sched)
    tpot_base = compute_tpot_stats(baseline)

    pcie_events_sched: list[dict] = []
    pcie_events_base: list[dict] = []
    if (base / "pcie_events_pcie_sched.json").exists():
        with open(base / "pcie_events_pcie_sched.json") as f:
            pcie_events_sched = json.load(f)
    if (base / "pcie_events_baseline.json").exists():
        with open(base / "pcie_events_baseline.json") as f:
            pcie_events_base = json.load(f)

    pcie_stats_sched = analyze_pcie_events(pcie_events_sched)
    pcie_stats_base = analyze_pcie_events(pcie_events_base)

    pcie_events_suspicious_duplicate = bool(
        pcie_events_sched
        and pcie_events_base
        and json.dumps(pcie_events_sched, sort_keys=True)
        == json.dumps(pcie_events_base, sort_keys=True)
    )

    # Extract PCIe scheduler stats from logs
    scheduler_stats_sched = extract_pcie_scheduler_stats(base / "vllm_state_pcie_sched.log")
    scheduler_stats_base = extract_pcie_scheduler_stats(base / "vllm_state_baseline.log")

    # Prefetch cached stats
    def cached_stats(rows: list[dict]) -> dict:
        hit = [r for r in rows if r.get("cached_tokens", 0) or r.get("prefetch_cached_tokens", 0)]
        return {"hit_count": len(hit), "total": len(rows), "hit_rate_pct": len(hit) / len(rows) * 100 if rows else 0}

    cached_sched = cached_stats(pcie_sched)
    cached_base = cached_stats(baseline)

    # 按类别分组配置，便于阅读
    def group_config(cfg: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
        groups: dict[str, list[tuple[str, str]]] = {
            "系统/vLLM": [],
            "实验": [],
            "数据集": [],
            "其他": [],
        }
        for k, v in sorted(cfg.items()):
            if not k or not v:
                continue
            if k.startswith(("MODEL_", "GPU_", "VLLM_", "KV_", "SWAP_", "API_", "PCIE_", "NUM_GPU_BLOCKS")):
                groups["系统/vLLM"].append((k, v))
            elif k.startswith(("QPS", "SEED", "PREFETCH_", "SCHEDULE_", "TIMEOUT", "REQUEST_", "TB_", "ENABLE_", "RESULTS_")):
                groups["实验"].append((k, v))
            elif k.startswith(("DATASET", "TRACE", "FULL_TRACE", "NUM_CONV", "MAX_INPUT", "DATA_ROOT")):
                groups["数据集"].append((k, v))
            else:
                groups["其他"].append((k, v))
        return groups

    cfg_groups = group_config(config)

    # Build report
    lines = [
        "# PCIe Scheduling A/B 实验报告",
        "",
        "## 1. 配置快照",
        "",
    ]
    for group_name, items in cfg_groups.items():
        if items:
            lines.append(f"### {group_name}")
            lines.append("")
            lines.append("| 参数 | 值 |")
            lines.append("|------|-----|")
            for k, v in items:
                lines.append(f"| {k} | {v} |")
            lines.append("")
    # 若 config 为空，补充 report 参数
    if not config:
        lines.extend([
            "| 参数 | 值 |",
            "|------|-----|",
            f"| DATASET | {args.dataset or 'N/A'} |",
            f"| QPS | {args.qps or 'N/A'} |",
            f"| LEAD_TIME | {args.lead_time or 'N/A'} |",
            "",
        ])

    lines.extend([
        "## 2. TTFT 对比",
        "",
        "| 指标 | PCIe Sched | Baseline | 变化 |",
        "|------|------------|----------|------|",
    ])

    if ttft_pcie and ttft_base:
        for k in ["mean", "p50", "p90", "p95", "p99"]:
            v_s = ttft_pcie.get(k, 0)
            v_b = ttft_base.get(k, 0)
            if v_b > 0:
                delta = (1 - v_s / v_b) * 100
                lines.append(f"| {k.capitalize()} (ms) | {v_s:.2f} | {v_b:.2f} | {delta:+.1f}% |")
            else:
                lines.append(f"| {k.capitalize()} (ms) | {v_s:.2f} | - | - |")
        if ttft_pcie.get("std") and ttft_base.get("std"):
            lines.append(f"| Std | {ttft_pcie['std']:.2f} | {ttft_base['std']:.2f} | - |")
    else:
        lines.append("| (无足够成功样本) | - | - | - |")

    lines.extend([
        "",
        "## 3. TPOT 对比",
        "",
        "| 指标 | PCIe Sched | Baseline |",
        "|------|------------|----------|",
    ])
    if tpot_pcie and tpot_base:
        for k in ["mean", "p50"]:
            lines.append(f"| {k.capitalize()} (ms) | {tpot_pcie.get(k, 0):.2f} | {tpot_base.get(k, 0):.2f} |")
    else:
        lines.append("| (无数据) | - | - |")

    lines.extend([
        "",
        "## 4. Prefetch 行为",
        "",
        "| 指标 | PCIe Sched | Baseline |",
        "|------|------------|----------|",
        f"| 成功样本数 | {cached_sched['total']} | {cached_base['total']} |",
        f"| Cached 命中数 | {cached_sched['hit_count']} | {cached_base['hit_count']} |",
        f"| 命中率 (%) | {cached_sched['hit_rate_pct']:.1f} | {cached_base['hit_rate_pct']:.1f} |",
        "",
        "## 5. PCIe 带宽统计 (若已采集)",
        "",
    ])

    if pcie_events_suspicious_duplicate:
        lines.extend([
            "⚠️ **警告**: 两侧 `pcie_events_*.json` 合并结果完全一致，第五部分统计必然相同。",
            "常见原因：Phase 之间未清空 profiler 目录中的 `pcie_events_*.json`，或采集前未调用 `stop_profile` 导致仍读到上一轮落盘数据。",
            "请使用已修复的 `run_pcie_scheduling_ab.sh`（含 `start_profile` / `stop_profile` 与 Phase 间删除事件文件）重新跑实验。",
            "",
        ])

    if pcie_stats_sched or pcie_stats_base:
        lines.extend([
            "| 指标 | PCIe Sched | Baseline |",
            "|------|------------|----------|",
        ])
        for k in ["h2d_count", "h2d_total_gb", "d2h_count", "d2h_total_gb", "h2d_bandwidth_gbps", "duration_s"]:
            v_s = pcie_stats_sched.get(k, "-")
            v_b = pcie_stats_base.get(k, "-")
            if isinstance(v_s, float):
                v_s = f"{v_s:.2f}"
            if isinstance(v_b, float):
                v_b = f"{v_b:.2f}"
            lines.append(f"| {k} | {v_s} | {v_b} |")
    else:
        lines.append("*未找到 pcie_events_*.json，跳过 PCIe 带宽统计*")

    lines.extend([
        "",
        "## 6. 稳定性",
        "",
        f"- PCIe Sched 成功样本: {ttft_pcie.get('count', 0)}",
        f"- Baseline 成功样本: {ttft_base.get('count', 0)}",
        "",
    ])

    # Add PCIe scheduler statistics section
    if scheduler_stats_sched:
        lines.extend([
            "## 7. PCIe Scheduler 统计",
            "",
            "### PCIe Sched 模式",
            "",
            "| 指标 | 值 | 说明 |",
            "|------|-----|------|",
            f"| 总提交数 (submitted) | {scheduler_stats_sched.get('submitted', 0)} | 所有传输请求 |",
            f"| Prefetch 延迟数 (deferred) | {scheduler_stats_sched.get('deferred', 0)} | 因 block 不足被延迟 |",
            f"| Restore 调度数 | {scheduler_stats_sched.get('restore', 0)} | 高优先级恢复 |",
            f"| Prefetch 调度数 | {scheduler_stats_sched.get('prefetch', 0)} | 中优先级预取 |",
            f"| Evict 调度数 | {scheduler_stats_sched.get('evict', 0)} | 低优先级驱逐 |",
            f"| 最大队列深度 (max_queue) | {scheduler_stats_sched.get('max_queue', 0)} | 挂起传输峰值 |",
            f"| H2D 限流次数 (throttled) | {scheduler_stats_sched.get('throttled', 0)} | 并发达上限 |",
            f"| PP 空闲期冲刷次数 | {scheduler_stats_sched.get('pp_idle_flushes', 0)} | PP 感知调度 |",
            f"| Prefetch 防饥饿分发 | {scheduler_stats_sched.get('prefetch_starved', 0)} | 超 max_queue_wait_ms 强制出队 |",
            "",
        ])
        if scheduler_stats_sched.get('deferred', 0) > 0:
            lines.append(f"**🎯 Prefetch 降优先级生效**: {scheduler_stats_sched['deferred']} 次被延迟")
            lines.append("")
        if scheduler_stats_sched.get('throttled', 0) > 0:
            lines.append(f"**⏸️ H2D 并发控制生效**: {scheduler_stats_sched['throttled']} 次因并发限制被限流")
            lines.append("")
    elif scheduler_stats_base:
        lines.extend([
            "## 7. PCIe Scheduler 统计",
            "",
            "⚠️ PCIe Sched 模式未找到调度器统计，但 Baseline 中有（配置可能有误）",
            "",
        ])
    else:
        lines.extend([
            "## 7. PCIe Scheduler 统计",
            "",
            "*未找到 PCIe Scheduler 统计日志*",
            "",
        ])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
