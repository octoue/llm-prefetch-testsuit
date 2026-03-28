#!/usr/bin/env python3
"""
PCIe 实验可视化脚本：Timeline Gantt Chart、H2D Latency CDF、TTFT Boxplot。

用法:
  # Timeline + CDF（对比 Baseline vs Sched）
  python plot_pcie_timeline.py \
    --sched results/.../pcie_events_pcie_sched.json \
    --baseline results/.../pcie_events_baseline.json \
    --window-start 50000 --window-end 50500 \
    --output figures/

  # TTFT Boxplot（多组 JSONL）
  python plot_pcie_timeline.py \
    --ttft-files "G3:results/.../prefetch_g3.jsonl" "G1:results/.../prefetch_g1.jsonl" \
    --output figures/

  # Ablation 柱状图
  python plot_pcie_timeline.py \
    --ablation-files "G0:r/g0.jsonl" "G1:r/g1.jsonl" "G2:r/g2.jsonl" "G3:r/g3.jsonl" \
    --output figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ============================================================
# Color scheme
# ============================================================
COLOR_MAP = {
    "Prefetch": "#2ecc71",
    "Restore": "#e67e22",
    "Evict": "#95a5a6",
    "PP_P2P_Send": "#3498db",
    "PP_P2P_Recv": "#2c3e50",
    "PP_TP_AllGather_Reconstruct": "#9b59b6",
    "PP_Transfer": "#3498db",
}

TRACK_LABELS = [
    "GPU0-H2D", "GPU0-D2H", "GPU0-P2P",
    "GPU1-H2D", "GPU1-D2H", "GPU1-P2P",
]

DIRECTION_MAP = {"H2D": 0, "D2H": 1, "P2P": 2}


def load_events(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def infer_direction(e: dict) -> str:
    if "direction" in e:
        return e["direction"]
    op = e.get("op_type", "")
    if op in ("Prefetch", "Restore"):
        return "H2D"
    if op == "Evict":
        return "D2H"
    return "P2P"


# ============================================================
# Chart A: PCIe Transfer Timeline (dual-panel Gantt)
# ============================================================
def plot_timeline(events: list[dict], ax: plt.Axes, title: str,
                  t_start: float, t_end: float):
    """Draw a Gantt chart on the given axes."""
    for e in events:
        start_ms = e.get("start_ms", e["start_us"] / 1000.0)
        dur = e.get("duration_ms", (e["end_us"] - e["start_us"]) / 1000.0)
        rel_start = start_ms - t_start
        if rel_start + dur < 0 or rel_start > (t_end - t_start):
            continue

        gpu = e.get("gpu_id", 0)
        direction = infer_direction(e)
        d_idx = DIRECTION_MAP.get(direction, 2)
        y = gpu * 3 + d_idx
        color = COLOR_MAP.get(e["op_type"], "#bdc3c7")
        ax.barh(y, dur, left=rel_start, height=0.8, color=color, alpha=0.85,
                edgecolor="none")

    ax.set_yticks(range(len(TRACK_LABELS)))
    ax.set_yticklabels(TRACK_LABELS, fontsize=8)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, t_end - t_start)
    ax.invert_yaxis()


def chart_timeline(baseline_path: str, sched_path: str,
                   t_start: float, t_end: float, output_dir: str):
    baseline = load_events(baseline_path)
    sched = load_events(sched_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    plot_timeline(baseline, ax1, "Baseline (no scheduling)", t_start, t_end)
    plot_timeline(sched, ax2, "PCIe Scheduling", t_start, t_end)

    # Legend
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in COLOR_MAP.items()]
    fig.legend(handles=patches, loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    out = os.path.join(output_dir, "pcie_timeline.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Chart B: Per-Transfer H2D Latency CDF
# ============================================================
def chart_h2d_cdf(baseline_path: str, sched_path: str, output_dir: str):
    def extract_h2d_durations(path: str) -> np.ndarray:
        events = load_events(path)
        durs = [e["duration_ms"] for e in events
                if e.get("op_type") in ("Prefetch", "Restore")]
        return np.array(durs) if durs else np.array([])

    bl_durs = extract_h2d_durations(baseline_path)
    sc_durs = extract_h2d_durations(sched_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    for durs, label, color in [
        (bl_durs, "Baseline", "#e74c3c"),
        (sc_durs, "PCIe Sched", "#2ecc71"),
    ]:
        if len(durs) == 0:
            continue
        sorted_d = np.sort(durs)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.plot(sorted_d, cdf, label=f"{label} (n={len(durs)})", color=color,
                linewidth=1.5)

    ax.set_xlabel("H2D Transfer Duration (ms)", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_title("Per-Transfer H2D Latency CDF (Prefetch + Restore)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(output_dir, "h2d_latency_cdf.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Chart C: TTFT Boxplot (multi-group)
# ============================================================
def load_ttft(path: str) -> np.ndarray:
    ttfts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            val = d.get("ttft_ms")
            if val is not None:
                ttfts.append(val)
    return np.array(ttfts) if ttfts else np.array([])


def chart_ttft_boxplot(labeled_files: list[str], output_dir: str):
    """labeled_files: list of 'Label:path.jsonl' strings."""
    groups = []
    for item in labeled_files:
        if ":" not in item:
            print(f"Warning: skipping invalid --ttft-files entry: {item}")
            continue
        label, path = item.split(":", 1)
        ttfts = load_ttft(path)
        if len(ttfts) > 0:
            groups.append((label, ttfts))

    if not groups:
        print("No valid TTFT data found.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.5), 5))
    data = [g[1] for g in groups]
    labels = [g[0] for g in groups]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    showfliers=True, flierprops=dict(marker=".", markersize=3))

    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#9b59b6"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    ax.set_ylabel("TTFT (ms)", fontsize=10)
    ax.set_title("TTFT Distribution by Group", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add median text
    for i, (label, vals) in enumerate(groups):
        med = np.median(vals)
        ax.text(i + 1, med, f" {med:.0f}", va="center", fontsize=7, color="red")

    out = os.path.join(output_dir, "ttft_boxplot.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Chart D: Ablation bar chart
# ============================================================
def chart_ablation_bar(labeled_files: list[str], output_dir: str):
    """Bar chart comparing mean/P50/P95 TTFT across ablation groups."""
    groups = []
    for item in labeled_files:
        if ":" not in item:
            continue
        label, path = item.split(":", 1)
        ttfts = load_ttft(path)
        if len(ttfts) > 0:
            groups.append((label, ttfts))

    if not groups:
        print("No valid ablation data found.")
        return

    labels = [g[0] for g in groups]
    means = [np.mean(g[1]) for g in groups]
    p50s = [np.percentile(g[1], 50) for g in groups]
    p95s = [np.percentile(g[1], 95) for g in groups]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 2), 5))
    ax.bar(x - width, means, width, label="Mean", color="#3498db", alpha=0.8)
    ax.bar(x, p50s, width, label="P50", color="#2ecc71", alpha=0.8)
    ax.bar(x + width, p95s, width, label="P95", color="#e74c3c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("TTFT (ms)", fontsize=10)
    ax.set_title("Ablation: TTFT by Group", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels
    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=7)

    out = os.path.join(output_dir, "ablation_ttft.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="PCIe experiment visualization")
    parser.add_argument("--sched", help="PCIe events JSON (scheduled)")
    parser.add_argument("--baseline", help="PCIe events JSON (baseline)")
    parser.add_argument("--window-start", type=float, default=0,
                        help="Timeline window start (ms, absolute)")
    parser.add_argument("--window-end", type=float, default=0,
                        help="Timeline window end (ms, absolute). "
                             "0=auto (first 500ms of activity)")
    parser.add_argument("--ttft-files", nargs="*",
                        help="'Label:path.jsonl' pairs for TTFT boxplot")
    parser.add_argument("--ablation-files", nargs="*",
                        help="'Label:path.jsonl' pairs for ablation bar chart")
    parser.add_argument("--output", default="figures/",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Auto-detect window if not specified
    if args.sched and args.baseline:
        if args.window_end <= args.window_start:
            events = load_events(args.sched) + load_events(args.baseline)
            if events:
                all_starts = [e.get("start_ms", e["start_us"] / 1000.0)
                              for e in events]
                t_min = min(all_starts)
                args.window_start = t_min
                args.window_end = t_min + 500  # 500ms window
                print(f"Auto window: {args.window_start:.1f} - "
                      f"{args.window_end:.1f} ms")

        chart_timeline(args.baseline, args.sched,
                       args.window_start, args.window_end, args.output)
        chart_h2d_cdf(args.baseline, args.sched, args.output)

    if args.ttft_files:
        chart_ttft_boxplot(args.ttft_files, args.output)

    if args.ablation_files:
        chart_ablation_bar(args.ablation_files, args.output)

    if not args.sched and not args.baseline and not args.ttft_files \
            and not args.ablation_files:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
