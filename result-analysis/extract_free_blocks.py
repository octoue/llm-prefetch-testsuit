#!/usr/bin/env python3
"""
Extract free_blocks time series from vLLM logs and generate comparison figures.

Parses lines like:
  (EngineCore_DP0 pid=...) INFO 04-01 01:41:17 [scheduler.py:840] Request ... free_blocks=253

Usage:
  # Auto-detect from experiment directory:
  python extract_free_blocks.py --exp-dir /path/to/ablation_result_dir/ --output figures/

  # Specify logs manually with time ranges (HH:MM:SS):
  python extract_free_blocks.py \
    --logs "G1:vllm_log_g1_g0.log:01:40:00-01:46:30" \
         "G0:vllm_log_g1_g0.log:01:47:00-01:54:30" \
         "G3:vllm_log_g3.log" \
    --output figures/
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
})

GROUP_COLORS = {
    "G0": "#95a5a6",
    "G1": "#e74c3c",
    "G2": "#3498db",
    "G3": "#2ecc71",
}
GROUP_LABELS = {
    "G0": "Baseline (no prefetch)",
    "G1": "Prefetch only (no scheduling)",
    "G2": "Scheduler (no phase-aware)",
    "G3": "Full system",
}

# Regex for the free_blocks log line
# Example: (EngineCore_DP0 pid=1273775) INFO 04-01 01:41:17 [scheduler.py:840] Request ... free_blocks=253
FREE_BLOCKS_RE = re.compile(
    r"INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+.*free_blocks=(\d+)"
)
TOKENS_RE = re.compile(r"loading\s+(\d+)\s+external\s+tokens")


def parse_time(time_str: str) -> datetime:
    """Parse 'MM-DD HH:MM:SS' to datetime (year doesn't matter for relative time)."""
    return datetime.strptime(f"2026-{time_str}", "%Y-%m-%d %H:%M:%S")


def parse_hhmmss(s: str) -> datetime:
    """Parse 'HH:MM:SS' to datetime."""
    return datetime.strptime(f"2026-01-01 {s}", "%Y-%m-%d %H:%M:%S")


def extract_free_blocks(log_path: str,
                        time_start: str | None = None,
                        time_end: str | None = None) -> list[dict]:
    """Extract (timestamp, free_blocks, tokens_loading) from a vLLM log file."""
    entries = []
    t_start = parse_hhmmss(time_start) if time_start else None
    t_end = parse_hhmmss(time_end) if time_end else None

    with open(log_path) as f:
        for line in f:
            m = FREE_BLOCKS_RE.search(line)
            if not m:
                continue
            ts = parse_time(m.group(1))
            free_blocks = int(m.group(2))

            # Filter by time range
            ts_hms = ts.replace(year=2026, month=1, day=1)
            if t_start and ts_hms < t_start:
                continue
            if t_end and ts_hms > t_end:
                continue

            tokens = 0
            tm = TOKENS_RE.search(line)
            if tm:
                tokens = int(tm.group(1))

            entries.append({
                "timestamp": ts,
                "free_blocks": free_blocks,
                "tokens_loading": tokens,
            })

    return entries


def auto_detect_time_ranges(exp_dir: str) -> dict:
    """Detect time ranges for each group from JSONL result files."""
    d = Path(exp_dir)
    ranges = {}

    for pattern, group in [
        ("prefetch_g0_*", "G0"), ("prefetch_g1_*", "G1"),
        ("prefetch_g2_*", "G2"), ("prefetch_g3_*", "G3"),
    ]:
        matches = [m for m in d.glob(pattern) if m.suffix == ".jsonl"]
        if not matches:
            continue
        timestamps = []
        with open(matches[0]) as f:
            for line in f:
                data = json.loads(line)
                if "timestamp" in data:
                    timestamps.append(data["timestamp"])
        if timestamps:
            from datetime import timezone
            t_min = datetime.fromtimestamp(min(timestamps))
            t_max = datetime.fromtimestamp(max(timestamps))
            # Add 30s margin on each side
            from datetime import timedelta
            t_min -= timedelta(seconds=60)
            t_max += timedelta(seconds=60)
            ranges[group] = {
                "start": t_min.strftime("%H:%M:%S"),
                "end": t_max.strftime("%H:%M:%S"),
            }

    return ranges


def auto_detect_logs(exp_dir: str) -> dict:
    """Detect which log file to use for each group."""
    d = Path(exp_dir)
    logs = {}

    # G0 and G1 share vllm_log_g1_g0.log
    g1_g0 = d / "vllm_log_g1_g0.log"
    if g1_g0.exists():
        logs["G0"] = str(g1_g0)
        logs["G1"] = str(g1_g0)

    for g, name in [("G2", "vllm_log_g2.log"), ("G3", "vllm_log_g3.log")]:
        p = d / name
        if p.exists():
            logs[g] = str(p)

    return logs


def figure_free_blocks(group_data: dict[str, list[dict]], output_dir: str,
                       threshold: int = 150, total_blocks: int = 1000):
    """Generate free_blocks time series comparison figure.

    Panel (a): Time series of free_blocks for each group
    Panel (b): Distribution of free_blocks (histogram or CDF)
    """
    if not group_data:
        print("No free_blocks data found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Panel (a): Time series ---
    for group in sorted(group_data.keys()):
        entries = group_data[group]
        if not entries:
            continue
        # Normalize time to start from 0
        t0 = entries[0]["timestamp"]
        times_sec = [(e["timestamp"] - t0).total_seconds() for e in entries]
        blocks = [e["free_blocks"] for e in entries]
        color = GROUP_COLORS.get(group, "#333")
        label = GROUP_LABELS.get(group, group)
        ax1.plot(times_sec, blocks, color=color, alpha=0.7, linewidth=1.0,
                 label=f"{label} (n={len(entries)})", zorder=2)

    # Danger zone
    ax1.axhline(y=threshold, color="#e74c3c", linestyle="--", linewidth=1.0,
                alpha=0.7, zorder=1)
    ax1.text(ax1.get_xlim()[1] * 0.02, threshold + 15,
             f"Prefetch threshold ({threshold})", fontsize=8, color="#e74c3c")
    ax1.axhspan(0, threshold, alpha=0.05, color="#e74c3c", zorder=0)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Free GPU Blocks")
    ax1.set_title("(a) GPU Memory Pressure Over Time")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(bottom=0)

    # --- Panel (b): CDF of free_blocks ---
    for group in sorted(group_data.keys()):
        entries = group_data[group]
        if not entries:
            continue
        blocks = np.array([e["free_blocks"] for e in entries])
        sorted_b = np.sort(blocks)
        cdf = np.arange(1, len(sorted_b) + 1) / len(sorted_b)
        color = GROUP_COLORS.get(group, "#333")
        label = GROUP_LABELS.get(group, group)
        ax2.plot(sorted_b, cdf, color=color, linewidth=1.8, label=label)

    ax2.axvline(x=threshold, color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.text(threshold + 5, 0.5, f"Threshold\n({threshold})", fontsize=8,
             color="#e74c3c", va="center")

    ax2.set_xlabel("Free GPU Blocks")
    ax2.set_ylabel("CDF")
    ax2.set_title("(b) Free Blocks Distribution")
    ax2.legend(loc="lower right", fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(f"GPU Block Pressure (total={total_blocks}, threshold={threshold})",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    out = os.path.join(output_dir, "m4_free_blocks.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[M4] Saved: {out}")

    # Statistics
    print("[M4] Free Blocks Statistics:")
    for group in sorted(group_data.keys()):
        entries = group_data[group]
        if not entries:
            continue
        blocks = np.array([e["free_blocks"] for e in entries])
        below = np.sum(blocks < threshold) / len(blocks) * 100
        print(f"  {group}: n={len(entries)}, mean={np.mean(blocks):.0f}, "
              f"P5={np.percentile(blocks, 5):.0f}, P50={np.median(blocks):.0f}, "
              f"min={np.min(blocks)}, below_threshold={below:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Extract free_blocks from vLLM logs and generate figures")
    parser.add_argument("--exp-dir",
                        help="Experiment result directory (auto-detect)")
    parser.add_argument("--logs", nargs="*",
                        help="'Group:logfile[:HH:MM:SS-HH:MM:SS]' entries")
    parser.add_argument("--threshold", type=int, default=150,
                        help="Prefetch block threshold (default: 150)")
    parser.add_argument("--total-blocks", type=int, default=1000,
                        help="Total GPU blocks (default: 1000)")
    parser.add_argument("--output", default="figures/motivation/",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    group_data = {}

    if args.exp_dir:
        log_files = auto_detect_logs(args.exp_dir)
        time_ranges = auto_detect_time_ranges(args.exp_dir)
        print(f"Auto-detected in {args.exp_dir}:")
        for g in sorted(log_files.keys()):
            tr = time_ranges.get(g, {})
            print(f"  {g}: {Path(log_files[g]).name} "
                  f"[{tr.get('start', '?')}-{tr.get('end', '?')}]")

        for group, log_path in log_files.items():
            tr = time_ranges.get(group, {})
            entries = extract_free_blocks(
                log_path,
                time_start=tr.get("start"),
                time_end=tr.get("end"),
            )
            if entries:
                group_data[group] = entries

    if args.logs:
        for spec in args.logs:
            parts = spec.split(":")
            if len(parts) < 2:
                continue
            group = parts[0]
            log_path = parts[1]
            t_start = t_end = None
            if len(parts) >= 3:
                time_range = parts[2]
                if "-" in time_range:
                    t_start, t_end = time_range.split("-", 1)
            entries = extract_free_blocks(log_path, t_start, t_end)
            if entries:
                group_data[group] = entries

    if group_data:
        figure_free_blocks(group_data, args.output,
                           threshold=args.threshold,
                           total_blocks=args.total_blocks)
    else:
        print("No data extracted. Check log paths and time ranges.")


if __name__ == "__main__":
    main()
