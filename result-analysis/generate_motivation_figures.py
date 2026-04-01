#!/usr/bin/env python3
"""
Motivation Section Figure Generator for PCIe Scheduling Paper.
Publication-quality figures for OSDI/SOSP style.

Generates 3 figures (G1-only, problem characterization):
  Fig1: PCIe Contention & Priority Inversion (stacked area + bar chart)
  Fig2: TTFT Long-Tail CDF (Turn 1 vs Turn 2-5)
  Fig3: PCIe Utilization Heatmap (full trace)

Usage:
  python generate_motivation_figures.py \
    --exp-dir ablation-results-heavy/20260401_011635_*/ \
    --output claude-docs/motivation-figures/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ============================================================
# Global Style — OSDI/SOSP publication quality
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,       # embed fonts in PDF
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
})

# Color semantics (per task_draw_motivation_pic.md)
C_RESTORE = "#C0392B"   # red/dark orange — critical path
C_PREFETCH = "#2980B9"  # blue — non-critical background
C_EVICT = "#95A5A6"     # gray
C_PP_RECV = "#BDC3C7"   # light gray — PP communication background
C_WARN = "#E74C3C"      # red — warning/alert
C_OK = "#27AE60"        # green — correct/good


# ============================================================
# Data loading
# ============================================================
def load_events(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else [data]
    for e in events:
        if "end_us" not in e:
            e["end_us"] = e["start_us"] + e["duration_ms"] * 1000
        if "start_ms" not in e:
            e["start_ms"] = e["start_us"] / 1000.0
    return events


def load_ttft(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("ttft_ms") is not None and d.get("success", True):
                records.append(d)
    return records


# ============================================================
# Helpers
# ============================================================
def _find_priority_inversion_pairs(events: list[dict], window_us: int = 50_000):
    """Find (prefetch, restore) pairs where prefetch started first within window."""
    restores = sorted([e for e in events if e["op_type"] == "Restore"],
                      key=lambda x: x["start_us"])
    prefetches = sorted([e for e in events if e["op_type"] == "Prefetch"],
                        key=lambda x: x["start_us"])

    prefetch_first = []
    restore_first = []
    pf_idx = 0
    for r in restores:
        while pf_idx < len(prefetches) and prefetches[pf_idx]["start_us"] < r["start_us"] - window_us:
            pf_idx += 1
        j = pf_idx
        while j < len(prefetches) and prefetches[j]["start_us"] <= r["start_us"] + window_us:
            p = prefetches[j]
            if p.get("gpu_id") == r.get("gpu_id"):
                if p["start_us"] < r["start_us"]:
                    prefetch_first.append((p, r))
                else:
                    restore_first.append((r, p))
            j += 1
    return prefetch_first, restore_first


def _find_best_inversion_example(events: list[dict], prefetch_first: list):
    """Find the best priority inversion example for visualization.

    Prefer cases with:
    1. Moderate gap (3-15ms) — visible but not extreme
    2. Both transfers have substantial duration
    3. Clear overlap period
    """
    candidates = []
    for p, r in prefetch_first:
        gap_ms = (r["start_us"] - p["start_us"]) / 1000.0
        overlap_start = max(p["start_us"], r["start_us"])
        overlap_end = min(p["end_us"], r["end_us"])
        overlap_ms = max(0, (overlap_end - overlap_start) / 1000.0)
        if 2 < gap_ms < 15 and p["duration_ms"] > 5 and r["duration_ms"] > 5 and overlap_ms > 3:
            candidates.append((gap_ms, overlap_ms, p, r))

    if candidates:
        # Pick one with good visual balance
        candidates.sort(key=lambda x: x[1], reverse=True)  # most overlap
        return candidates[0][2], candidates[0][3]

    # Fallback: pick by largest gap with overlap
    for p, r in sorted(prefetch_first, key=lambda pr: pr[1]["start_us"] - pr[0]["start_us"], reverse=True):
        if min(p["end_us"], r["end_us"]) > max(p["start_us"], r["start_us"]):
            return p, r
    return prefetch_first[0] if prefetch_first else (None, None)


# ============================================================
# Figure 1: PCIe Contention & Priority Inversion
# ============================================================
def figure_1(trace_path: str, output_dir: str):
    """Fig1: Stacked area timeline + macro statistics.

    Panel (a): Stacked area chart showing bandwidth contention during
               a priority inversion event (~100ms window).
    Panel (b): Bar chart showing scheduling order stats + traffic volume.
    """
    events = load_events(trace_path)
    prefetch_first, restore_first = _find_priority_inversion_pairs(events)
    pf_count = len(prefetch_first)
    rf_count = len(restore_first)
    total = pf_count + rf_count

    print(f"[Fig1] Priority inversion: {pf_count}/{total} prefetch-first "
          f"({pf_count/total*100:.1f}%)" if total > 0 else "[Fig1] No pairs")

    # Find best example
    pf_ex, rs_ex = _find_best_inversion_example(events, prefetch_first)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4),
                                      gridspec_kw={"width_ratios": [1.6, 1]},
                                      constrained_layout=True)

    # ===== Panel (a): Stacked Area Timeline =====
    if pf_ex and rs_ex:
        gpu_id = pf_ex.get("gpu_id", 1)
        # Define window: center on the inversion event, ~120ms total
        center_us = (pf_ex["start_us"] + rs_ex["end_us"]) / 2
        half_win = 60_000  # 60ms each side
        win_start_us = center_us - half_win
        win_end_us = center_us + half_win

        # Collect all H2D events on this GPU in window
        h2d_window = [
            e for e in events
            if e["op_type"] in ("Prefetch", "Restore")
            and e.get("gpu_id") == gpu_id
            and e["end_us"] > win_start_us and e["start_us"] < win_end_us
        ]
        pp_window = [
            e for e in events
            if e["op_type"] == "PP_P2P_Recv"
            and e.get("gpu_id") == gpu_id
            and e["end_us"] > win_start_us and e["start_us"] < win_end_us
        ]

        # Build stacked area: sample at 0.1ms resolution
        win_dur_ms = (win_end_us - win_start_us) / 1000.0
        dt = 0.1  # ms
        t_points = np.arange(0, win_dur_ms, dt)
        bw_prefetch = np.zeros_like(t_points)
        bw_restore = np.zeros_like(t_points)

        for e in h2d_window:
            rel_start = (e["start_us"] - win_start_us) / 1000.0
            rel_end = (e["end_us"] - win_start_us) / 1000.0
            bw = e.get("bandwidth_gbps", 0)
            if bw <= 0:
                bw = e.get("size_bytes", 0) / 1e9 / (e["duration_ms"] / 1000.0) if e["duration_ms"] > 0 else 0
            mask = (t_points >= rel_start) & (t_points < rel_end)
            if e["op_type"] == "Prefetch":
                bw_prefetch[mask] += bw
            else:
                bw_restore[mask] += bw

        # Draw PP_Recv as gray background bands
        for e in pp_window:
            rel_start = max(0, (e["start_us"] - win_start_us) / 1000.0)
            rel_end = min(win_dur_ms, (e["end_us"] - win_start_us) / 1000.0)
            ax_a.axvspan(rel_start, rel_end, alpha=0.12, color=C_PP_RECV, zorder=0)

        # Stacked area
        ax_a.fill_between(t_points, 0, bw_prefetch,
                          color=C_PREFETCH, alpha=0.7, label="Prefetch (non-critical)",
                          linewidth=0, zorder=1)
        ax_a.fill_between(t_points, bw_prefetch, bw_prefetch + bw_restore,
                          color=C_RESTORE, alpha=0.8, label="Restore (critical path)",
                          linewidth=0, zorder=2)

        # Draw arrival arrows at top
        arrow_y = max(np.max(bw_prefetch + bw_restore) * 1.05, 22)
        pf_rel = (pf_ex["start_us"] - win_start_us) / 1000.0
        rs_rel = (rs_ex["start_us"] - win_start_us) / 1000.0

        ax_a.annotate("", xy=(pf_rel, arrow_y * 0.85), xytext=(pf_rel, arrow_y * 1.05),
                      arrowprops=dict(arrowstyle="-|>", color=C_PREFETCH, lw=2))
        ax_a.text(pf_rel, arrow_y * 1.08, "Prefetch\narrives", ha="center", va="bottom",
                  fontsize=9, color=C_PREFETCH, fontweight="bold")

        ax_a.annotate("", xy=(rs_rel, arrow_y * 0.85), xytext=(rs_rel, arrow_y * 1.05),
                      arrowprops=dict(arrowstyle="-|>", color=C_RESTORE, lw=2))
        ax_a.text(rs_rel, arrow_y * 1.08, "Restore\narrives", ha="center", va="bottom",
                  fontsize=9, color=C_RESTORE, fontweight="bold")

        # Highlight the contention zone with hatching
        overlap_start = max(pf_ex["start_us"], rs_ex["start_us"])
        overlap_end = min(pf_ex["end_us"], rs_ex["end_us"])
        if overlap_end > overlap_start:
            os_rel = (overlap_start - win_start_us) / 1000.0
            oe_rel = (overlap_end - win_start_us) / 1000.0
            ax_a.axvspan(os_rel, oe_rel, alpha=0.15, facecolor=C_WARN,
                         hatch="///", edgecolor=C_WARN, linewidth=0.5, zorder=3,
                         label="Bandwidth contention")

        # Priority Inversion label
        gap_ms = (rs_ex["start_us"] - pf_ex["start_us"]) / 1000.0
        mid_x = (pf_rel + rs_rel) / 2
        ax_a.text(mid_x, arrow_y * 0.5,
                  f"Priority Inversion\n({gap_ms:.1f}ms delay)",
                  ha="center", va="center", fontsize=10, fontweight="bold",
                  color=C_WARN,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC",
                            edgecolor=C_WARN, alpha=0.9, linewidth=1.2))

        # PCIe peak line
        ax_a.axhline(y=25.6, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax_a.text(win_dur_ms * 0.99, 25.6, "PCIe Gen4 peak", ha="right", va="bottom",
                  fontsize=8, color="gray", alpha=0.7)

        ax_a.set_xlabel("Time (ms)")
        ax_a.set_ylabel("PCIe Bandwidth (GB/s)")
        ax_a.set_xlim(0, win_dur_ms)
        ax_a.set_ylim(0, arrow_y * 1.3)
        ax_a.set_title("(a) PCIe H2D Bandwidth During Priority Inversion")
        ax_a.legend(fontsize=9, loc="lower right",
                    framealpha=0.9, edgecolor="gray")
        ax_a.grid(True, alpha=0.2, axis="y")

        print(f"[Fig1a] Example: Prefetch {pf_ex['duration_ms']:.1f}ms, "
              f"Restore {rs_ex['duration_ms']:.1f}ms, gap {gap_ms:.1f}ms, GPU{gpu_id}")

    # ===== Panel (b): Macro Statistics =====
    # Traffic volume
    restore_all = [e for e in events if e["op_type"] == "Restore"]
    prefetch_all = [e for e in events if e["op_type"] == "Prefetch"]
    restore_gb = sum(e.get("wire_bytes", e["size_bytes"]) for e in restore_all) / 1e9
    prefetch_gb = sum(e.get("wire_bytes", e["size_bytes"]) for e in prefetch_all) / 1e9

    # Two groups of bars
    x = np.array([0, 1.5])  # positions for two groups
    width = 0.35

    # Group 1: Scheduling order (%)
    if total > 0:
        pf_pct = pf_count / total * 100
        rf_pct = rf_count / total * 100
    else:
        pf_pct = rf_pct = 0

    bar1_a = ax_b.bar(x[0] - width / 2, pf_pct, width,
                      color=C_WARN, alpha=0.85, label="Prefetch First")
    bar1_b = ax_b.bar(x[0] + width / 2, rf_pct, width,
                      color=C_OK, alpha=0.85, label="Restore First")

    # Group 2: Traffic volume (GB) — use twin axis
    ax_b2 = ax_b.twinx()
    bar2_a = ax_b2.bar(x[1] - width / 2, restore_gb, width,
                       color=C_RESTORE, alpha=0.85)
    bar2_b = ax_b2.bar(x[1] + width / 2, prefetch_gb, width,
                       color=C_PREFETCH, alpha=0.85)

    # Labels on bars
    ax_b.text(x[0] - width / 2, pf_pct + 1.5, f"{pf_pct:.0f}%",
              ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_WARN)
    ax_b.text(x[0] + width / 2, rf_pct + 1.5, f"{rf_pct:.0f}%",
              ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_OK)
    ax_b2.text(x[1] - width / 2, restore_gb + 1, f"{restore_gb:.0f}",
               ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_RESTORE)
    ax_b2.text(x[1] + width / 2, prefetch_gb + 1, f"{prefetch_gb:.0f}",
               ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_PREFETCH)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Scheduling\nOrder (%)", "Traffic\nVolume (GB)"], fontsize=10)
    ax_b.set_ylabel("Percentage (%)")
    ax_b2.set_ylabel("Data Volume (GB)")
    ax_b.set_ylim(0, 80)
    ax_b2.set_ylim(0, 90)
    ax_b.set_title("(b) Scheduling Disorder Statistics")
    ax_b.grid(True, alpha=0.2, axis="y")

    # Combined legend
    legend_elements = [
        mpatches.Patch(color=C_WARN, alpha=0.85, label="Prefetch first (inversion)"),
        mpatches.Patch(color=C_OK, alpha=0.85, label="Restore first (correct)"),
        mpatches.Patch(color=C_RESTORE, alpha=0.85, label="Restore volume"),
        mpatches.Patch(color=C_PREFETCH, alpha=0.85, label="Prefetch volume"),
    ]
    ax_b.legend(handles=legend_elements, fontsize=7.5, loc="upper right",
                framealpha=0.9, edgecolor="gray")

    # Annotation
    ax_b.text(0.5, 0.02,
              "Equal traffic, no priority differentiation",
              transform=ax_b.transAxes, ha="center", va="bottom",
              fontsize=9, style="italic", color="#555")

    out = os.path.join(output_dir, "fig1_pcie_contention.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[Fig1] Saved: {out}")


# ============================================================
# Figure 2: TTFT Long-Tail CDF
# ============================================================
def figure_2(ttft_path: str, output_dir: str):
    """Fig2: TTFT CDF — Turn 1 (cold start) vs Turn 2-5 (with Prefetch cache).

    The counter-intuitive finding: cached requests are SLOWER than cold starts.
    """
    records = load_ttft(ttft_path)
    if not records:
        print("[Fig2] No TTFT data, skipping.")
        return

    # Split by turn
    turn1 = np.array([r["ttft_ms"] / 1000.0 for r in records if r.get("turn", 1) == 1])
    turn2_5 = np.array([r["ttft_ms"] / 1000.0 for r in records
                        if 2 <= r.get("turn", 1) <= 5])

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), constrained_layout=True)

    # CDF: Turn 1 (cold start) — black dashed
    if len(turn1) > 0:
        sorted_t1 = np.sort(turn1)
        cdf_t1 = np.arange(1, len(sorted_t1) + 1) / len(sorted_t1)
        ax.plot(sorted_t1, cdf_t1, color="black", linestyle="--", linewidth=2,
                label=f"Turn 1 — Cold Start (n={len(turn1)})", zorder=4)

    # CDF: Turn 2-5 (cached) — red solid
    if len(turn2_5) > 0:
        sorted_t25 = np.sort(turn2_5)
        cdf_t25 = np.arange(1, len(sorted_t25) + 1) / len(sorted_t25)
        ax.plot(sorted_t25, cdf_t25, color=C_WARN, linestyle="-", linewidth=2.5,
                label=f"Turn 2-5 — With Prefetch Cache (n={len(turn2_5)})", zorder=3)

    # P50 horizontal line
    ax.axhline(y=0.5, color="gray", linestyle="-.", alpha=0.4, linewidth=0.8)
    ax.text(0.5, 0.51, "P50", fontsize=9, color="gray", alpha=0.6)

    # Threshold vertical lines at 10s and 20s
    for thresh_s in [10, 20]:
        ax.axvline(x=thresh_s, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # Annotate Turn 2-5 stats at thresholds
    if len(turn2_5) > 0:
        pct_gt10 = np.sum(turn2_5 > 10) / len(turn2_5) * 100
        pct_gt20 = np.sum(turn2_5 > 20) / len(turn2_5) * 100
        # y position on CDF at threshold
        y_at_10 = np.searchsorted(sorted_t25, 10) / len(sorted_t25)
        y_at_20 = np.searchsorted(sorted_t25, 20) / len(sorted_t25)

        ax.annotate(f"{pct_gt10:.1f}% > 10s",
                    xy=(10, y_at_10), xytext=(14, y_at_10 - 0.12),
                    fontsize=11, fontweight="bold", color=C_WARN,
                    arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.5),
                    zorder=5)
        ax.annotate(f"{pct_gt20:.1f}% > 20s",
                    xy=(20, y_at_20), xytext=(23, y_at_20 - 0.10),
                    fontsize=11, fontweight="bold", color=C_WARN,
                    arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.5),
                    zorder=5)

    # "Counter-intuitive" annotation — highlight that red line is RIGHT of black line
    if len(turn1) > 0 and len(turn2_5) > 0:
        # Place annotation between the two CDF curves near P50
        t1_p50 = np.median(turn1)
        t25_p50 = np.median(turn2_5)
        mid_x = (t1_p50 + t25_p50) / 2
        ax.annotate("Cached requests\nare SLOWER than\ncold starts",
                    xy=(mid_x, 0.5), xytext=(mid_x + 6, 0.35),
                    fontsize=10, fontweight="bold", color="#333",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5,
                                    connectionstyle="arc3,rad=0.2"),
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3CD",
                              edgecolor="#F0C36D", alpha=0.95),
                    zorder=5)

    ax.set_xlabel("TTFT (seconds)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9, edgecolor="gray")
    ax.grid(True, alpha=0.25)

    out = os.path.join(output_dir, "fig2_ttft_longtail.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[Fig2] Saved: {out}")

    # Print stats
    if len(turn1) > 0:
        print(f"[Fig2] Turn 1:   n={len(turn1)}, P50={np.median(turn1):.1f}s, "
              f"mean={np.mean(turn1):.1f}s")
    if len(turn2_5) > 0:
        print(f"[Fig2] Turn 2-5: n={len(turn2_5)}, P50={np.median(turn2_5):.1f}s, "
              f"mean={np.mean(turn2_5):.1f}s, "
              f">{10}s: {np.sum(turn2_5>10)/len(turn2_5)*100:.1f}%, "
              f">{20}s: {np.sum(turn2_5>20)/len(turn2_5)*100:.1f}%")


# ============================================================
# Figure 3: PCIe Utilization Heatmap
# ============================================================
def figure_3(trace_path: str, output_dir: str, bin_ms: float = 200.0):
    """Fig3: Full-trace PCIe utilization heatmap.

    Shows traffic density over the entire experiment, color-coded by type.
    Each row is a transfer type, X is time, color intensity is bandwidth.
    """
    events = load_events(trace_path)

    # Filter to GPU1 (the contention hotspot in PP=2)
    gpu1 = [e for e in events if e.get("gpu_id") == 1]
    if not gpu1:
        gpu1 = events  # fallback

    # Define transfer categories and their colormaps
    categories = [
        ("Restore", "Reds", [e for e in gpu1 if e["op_type"] == "Restore"]),
        ("Prefetch", "Blues", [e for e in gpu1 if e["op_type"] == "Prefetch"]),
        ("PP_Recv", "Greys", [e for e in gpu1 if e["op_type"] == "PP_P2P_Recv"]),
        ("Evict", "Greens", [e for e in gpu1 if e["op_type"] == "Evict"]),
    ]

    # Time range
    all_starts = [e["start_us"] for e in gpu1]
    all_ends = [e["end_us"] for e in gpu1]
    if not all_starts:
        print("[Fig3] No events, skipping.")
        return
    t_min = min(all_starts)
    t_max = max(all_ends)
    total_dur_ms = (t_max - t_min) / 1000.0

    # Create time bins
    n_bins = int(np.ceil(total_dur_ms / bin_ms))
    bin_edges_ms = np.arange(n_bins + 1) * bin_ms

    # Compute bandwidth per bin for each category
    heatmap_data = {}
    for cat_name, cmap_name, cat_events in categories:
        bw_per_bin = np.zeros(n_bins)
        for e in cat_events:
            rel_start_ms = (e["start_us"] - t_min) / 1000.0
            rel_end_ms = (e["end_us"] - t_min) / 1000.0
            size_gb = e.get("wire_bytes", e.get("size_bytes", 0)) / 1e9

            # Distribute bandwidth across bins
            bin_start = max(0, int(rel_start_ms / bin_ms))
            bin_end = min(n_bins - 1, int(rel_end_ms / bin_ms))
            dur_ms = rel_end_ms - rel_start_ms
            if dur_ms <= 0:
                continue
            bw_gbps = size_gb / (dur_ms / 1000.0)  # GB/s

            for b in range(bin_start, bin_end + 1):
                b_start = b * bin_ms
                b_end = (b + 1) * bin_ms
                overlap_start = max(rel_start_ms, b_start)
                overlap_end = min(rel_end_ms, b_end)
                if overlap_end > overlap_start:
                    frac = (overlap_end - overlap_start) / bin_ms
                    bw_per_bin[b] += bw_gbps * frac

        heatmap_data[cat_name] = (bw_per_bin, cmap_name)

    # Plot: one row per category
    active_cats = [(name, data, cmap) for name, (data, cmap) in heatmap_data.items()
                   if np.max(data) > 0]

    fig, axes = plt.subplots(len(active_cats), 1, figsize=(14, 1.5 * len(active_cats) + 0.8),
                             constrained_layout=True, sharex=True)
    if len(active_cats) == 1:
        axes = [axes]

    # Convert bin times to seconds for x-axis
    bin_centers_s = (bin_edges_ms[:-1] + bin_ms / 2) / 1000.0

    for idx, (cat_name, bw_data, cmap_name) in enumerate(active_cats):
        ax = axes[idx]
        # Reshape to 1 row for imshow-like display
        data_2d = bw_data.reshape(1, -1)

        vmax = max(np.percentile(bw_data[bw_data > 0], 95) if np.any(bw_data > 0) else 1, 0.1)
        im = ax.imshow(data_2d, aspect="auto", cmap=cmap_name,
                       vmin=0, vmax=vmax,
                       extent=[bin_centers_s[0], bin_centers_s[-1], -0.5, 0.5],
                       interpolation="nearest")
        ax.set_yticks([0])
        ax.set_yticklabels([cat_name], fontsize=11)
        ax.tick_params(axis="y", length=0)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("GB/s", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds)")
    axes[0].set_title("PCIe Transfer Density — GPU1 (Full Trace)")

    out = os.path.join(output_dir, "fig3_pcie_heatmap.pdf")
    fig.savefig(out)
    plt.close(fig)

    print(f"[Fig3] Saved: {out}")
    print(f"[Fig3] Trace duration: {total_dur_ms/1000:.1f}s, bins: {n_bins} x {bin_ms}ms")
    for name, (data, _) in heatmap_data.items():
        if np.max(data) > 0:
            print(f"  {name}: peak={np.max(data):.1f} GB/s, "
                  f"active bins={np.sum(data > 0)}/{n_bins}")


# ============================================================
# Auto-detect files
# ============================================================
def auto_detect_files(exp_dir: str) -> dict:
    d = Path(exp_dir)
    result = {}
    for pattern, key in [
        ("pcie_events_g1_*", "trace_g1"),
    ]:
        matches = list(d.glob(pattern))
        if matches:
            result[key] = str(matches[0])
    for pattern, key in [
        ("prefetch_g1_*", "ttft_g1"),
    ]:
        matches = [m for m in d.glob(pattern) if m.suffix == ".jsonl"]
        if matches:
            result[key] = str(matches[0])
    return result


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate motivation figures (publication quality)")
    parser.add_argument("--exp-dir",
                        help="Experiment result directory (auto-detect G1 files)")
    parser.add_argument("--g1-trace",
                        help="G1 PCIe trace JSON (override auto-detect)")
    parser.add_argument("--g1-ttft",
                        help="G1 TTFT JSONL (override auto-detect)")
    parser.add_argument("--figures", nargs="*", default=["fig1", "fig2", "fig3"],
                        help="Which figures to generate (default: fig1 fig2 fig3)")
    parser.add_argument("--output", default="figures/motivation/",
                        help="Output directory")
    parser.add_argument("--heatmap-bin", type=float, default=200.0,
                        help="Heatmap time bin size in ms (default: 200)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    figures = set(f.lower() for f in args.figures)

    files = {}
    if args.exp_dir:
        files = auto_detect_files(args.exp_dir)
        print(f"Auto-detected files in {args.exp_dir}:")
        for k, v in sorted(files.items()):
            print(f"  {k}: {Path(v).name}")

    g1_trace = args.g1_trace or files.get("trace_g1")
    g1_ttft = args.g1_ttft or files.get("ttft_g1")

    if not g1_trace:
        print("ERROR: No G1 trace found. Use --exp-dir or --g1-trace.", file=sys.stderr)
        sys.exit(1)

    if "fig1" in figures:
        print("\n" + "=" * 60)
        print("Generating Fig1: PCIe Contention & Priority Inversion")
        print("=" * 60)
        figure_1(g1_trace, args.output)

    if "fig2" in figures:
        if g1_ttft:
            print("\n" + "=" * 60)
            print("Generating Fig2: TTFT Long-Tail CDF")
            print("=" * 60)
            figure_2(g1_ttft, args.output)
        else:
            print("Fig2: No G1 TTFT file found, skipping.")

    if "fig3" in figures:
        print("\n" + "=" * 60)
        print("Generating Fig3: PCIe Utilization Heatmap")
        print("=" * 60)
        figure_3(g1_trace, args.output, bin_ms=args.heatmap_bin)


if __name__ == "__main__":
    main()
