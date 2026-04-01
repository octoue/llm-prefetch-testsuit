#!/usr/bin/env python3
"""
Motivation Section Figure Generator for PCIe Scheduling Paper.

Generates 4 figures for the motivation section:
  M1: H2D transfer latency CDF — overlapped vs. non-overlapped with PP communication
  M2: PCIe transfer timeline (Gantt) — showing IDLE windows and contention
  M3: TTFT CDF — unscheduled vs. scheduled (multi-group)
  M4: PCIe bandwidth utilization vs. H2D transfer latency scatter

Usage:
  # Use a single experiment directory (auto-detect g0/g1/g2/g3 files):
  python generate_motivation_figures.py --exp-dir /path/to/ablation_result_dir/ --output figures/

  # Or specify files explicitly:
  python generate_motivation_figures.py \
    --baseline-trace pcie_events_g1_prefetch_only.json \
    --sched-trace pcie_events_g3_full_sched.json \
    --ttft-files "G0:prefetch_g0.jsonl" "G1:prefetch_g1.jsonl" "G2:prefetch_g2.jsonl" "G3:prefetch_g3.jsonl" \
    --output figures/

  # Select which figures to generate:
  python generate_motivation_figures.py --exp-dir ... --figures m1 m2 m3 m4

  # M2 timeline window control:
  python generate_motivation_figures.py --exp-dir ... --figures m2 \
    --window-start 50000 --window-duration 200
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
})

COLOR_MAP = {
    "Prefetch": "#2ecc71",
    "Restore": "#e67e22",
    "Evict": "#95a5a6",
    "PP_P2P_Send": "#3498db",
    "PP_P2P_Recv": "#85c1e9",
    "PP_TP_AllGather_Reconstruct": "#9b59b6",
}

GROUP_COLORS = {
    "G0": "#95a5a6",
    "G1": "#e74c3c",
    "G2": "#3498db",
    "G3": "#2ecc71",
}

GROUP_LABELS = {
    "G0": "Baseline (no prefetch)",
    "G1": "Prefetch only",
    "G2": "Scheduler (no phase-aware)",
    "G3": "Full system",
}


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
    return events


def load_ttft(path: str) -> np.ndarray:
    ttfts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            val = d.get("ttft_ms")
            if val is not None and d.get("success", True):
                ttfts.append(val)
    return np.array(ttfts)


# ============================================================
# M1: H2D Traffic Characterization & Scheduling Opportunity
# ============================================================
def figure_m1(trace_paths: dict[str, str], output_dir: str):
    """M1: H2D traffic characterization across experimental groups.

    Panel (a): H2D transfer count and volume by group — shows Prefetch adds
               significant traffic that needs to be managed.
    Panel (b): Restore latency CDF across groups — shows the impact of
               uncoordinated traffic on critical-path transfers.
    """
    groups_data = {}
    for label, path in trace_paths.items():
        events = load_events(path)
        restore = [e for e in events if e["op_type"] == "Restore"]
        prefetch = [e for e in events if e["op_type"] == "Prefetch"]
        evict = [e for e in events if e["op_type"] == "Evict"]
        groups_data[label] = {
            "restore": restore,
            "prefetch": prefetch,
            "evict": evict,
            "restore_lats": np.array([e["duration_ms"] for e in restore]) if restore else np.array([]),
            "prefetch_lats": np.array([e["duration_ms"] for e in prefetch]) if prefetch else np.array([]),
            "restore_bytes": sum(e.get("wire_bytes", e["size_bytes"]) for e in restore),
            "prefetch_bytes": sum(e.get("wire_bytes", e["size_bytes"]) for e in prefetch),
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    groups_sorted = sorted(groups_data.keys())

    # --- Panel (a): Stacked bar chart of H2D traffic ---
    x = np.arange(len(groups_sorted))
    width = 0.35

    restore_counts = [len(groups_data[g]["restore"]) for g in groups_sorted]
    prefetch_counts = [len(groups_data[g]["prefetch"]) for g in groups_sorted]
    restore_gb = [groups_data[g]["restore_bytes"] / 1e9 for g in groups_sorted]
    prefetch_gb = [groups_data[g]["prefetch_bytes"] / 1e9 for g in groups_sorted]

    # Left bars: transfer count
    bars_r = ax1.bar(x - width / 2, restore_counts, width * 0.9,
                     label="Restore", color="#e67e22", alpha=0.8)
    bars_p = ax1.bar(x - width / 2, prefetch_counts, width * 0.9,
                     bottom=restore_counts, label="Prefetch", color="#2ecc71", alpha=0.8)

    # Right bars: volume (GB) on twin axis
    ax1_twin = ax1.twinx()
    total_gb = [r + p for r, p in zip(restore_gb, prefetch_gb)]
    ax1_twin.bar(x + width / 2, total_gb, width * 0.9,
                 color="#3498db", alpha=0.4, label="Total H2D (GB)")

    # Labels
    for i, (rc, pc) in enumerate(zip(restore_counts, prefetch_counts)):
        total = rc + pc
        if total > 0:
            ax1.text(x[i] - width / 2, total + 5, str(total),
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([GROUP_LABELS.get(g, g).replace(" ", "\n")
                         for g in groups_sorted], fontsize=7)
    ax1.set_ylabel("Transfer Count", color="#333")
    ax1_twin.set_ylabel("Total Volume (GB)", color="#3498db")
    ax1.set_title("(a) H2D Traffic Volume")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.2, axis="y")

    # --- Panel (b): H2D concurrency & PP overlap for G1 (the problem scenario) ---
    # Show: in G1 (prefetch only, no scheduler), how many H2D transfers
    # run concurrently? And how many overlap with PP_Recv?
    g1_key = "G1" if "G1" in trace_paths else None
    if g1_key:
        g1_events = load_events(trace_paths[g1_key])
        g1_h2d = sorted([e for e in g1_events if e["op_type"] in ("Prefetch", "Restore")],
                        key=lambda x: x["start_us"])
        g1_pp_recv = sorted([e for e in g1_events if e["op_type"] == "PP_P2P_Recv"],
                            key=lambda x: x["start_us"])

        # Compute per-H2D: concurrency count and PP overlap
        concurrency = []
        pp_overlap_flag = []  # True if overlaps with PP_Recv on same GPU
        pp_recv_by_gpu = defaultdict(list)
        for e in g1_pp_recv:
            pp_recv_by_gpu[e["gpu_id"]].append(e)

        for i, h in enumerate(g1_h2d):
            # Count concurrent H2D
            cnt = sum(1 for j, o in enumerate(g1_h2d)
                      if j != i and o["start_us"] < h["end_us"] and o["end_us"] > h["start_us"])
            concurrency.append(cnt)
            # Check PP overlap on same GPU
            gpu = h.get("gpu_id", 0)
            has_pp = any(p["end_us"] > h["start_us"] and p["start_us"] < h["end_us"]
                         for p in pp_recv_by_gpu.get(gpu, []))
            pp_overlap_flag.append(has_pp)

        concurrency = np.array(concurrency)
        pp_overlap_flag = np.array(pp_overlap_flag)

        # Stacked histogram: concurrency by PP overlap status
        max_conc = int(concurrency.max()) + 1
        bins = np.arange(-0.5, max_conc + 0.5, 1)
        conc_no_pp = concurrency[~pp_overlap_flag]
        conc_pp = concurrency[pp_overlap_flag]

        ax2.hist([conc_no_pp, conc_pp], bins=bins, stacked=True,
                 color=["#2ecc71", "#e74c3c"], alpha=0.7,
                 label=[f"IDLE window ({len(conc_no_pp)})",
                        f"During PP comm ({len(conc_pp)})"],
                 edgecolor="white", linewidth=0.5)

        ax2.set_xlabel("Concurrent H2D Transfers")
        ax2.set_ylabel("Count")
        ax2.set_title("(b) H2D Concurrency (Prefetch Only)")
        ax2.legend(fontsize=8)
        ax2.set_xticks(range(max_conc))
        ax2.grid(True, alpha=0.2, axis="y")

        # Annotate key stats
        pct_pp = np.sum(pp_overlap_flag) / len(pp_overlap_flag) * 100
        pct_conc = np.sum(concurrency > 0) / len(concurrency) * 100
        ax2.text(0.97, 0.95,
                 f"{pct_pp:.0f}% during PP\n{pct_conc:.0f}% concurrent",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "G1 trace not available", transform=ax2.transAxes,
                 ha="center", va="center")

    fig.suptitle("KV Cache Prefetch Increases H2D Traffic — Scheduling Required",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    out = os.path.join(output_dir, "m1_h2d_traffic.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[M1] Saved: {out}")

    # Print statistics
    print("[M1] H2D Traffic Summary:")
    for g in groups_sorted:
        d = groups_data[g]
        total = len(d["restore"]) + len(d["prefetch"])
        total_gb = (d["restore_bytes"] + d["prefetch_bytes"]) / 1e9
        rl = d["restore_lats"]
        print(f"  {g}: Restore={len(d['restore'])}, Prefetch={len(d['prefetch'])}, "
              f"Total H2D={total}, Volume={total_gb:.1f}GB" +
              (f", Restore P50={np.median(rl):.2f}ms, P95={np.percentile(rl, 95):.2f}ms"
               if len(rl) > 0 else ""))


# ============================================================
# M2: PCIe Transfer Timeline (Gantt)
# ============================================================
TRACK_ORDER = {
    # gpu_id -> direction -> track_index
    (0, "H2D"): 0, (0, "D2H"): 1, (0, "P2P"): 2,
    (1, "H2D"): 3, (1, "D2H"): 4, (1, "P2P"): 5,
}
TRACK_LABELS = [
    "GPU0 H2D", "GPU0 D2H", "GPU0 P2P",
    "GPU1 H2D", "GPU1 D2H", "GPU1 P2P",
]


def _infer_direction(e: dict) -> str:
    if "direction" in e:
        return e["direction"]
    op = e.get("op_type", "")
    if op in ("Prefetch", "Restore"):
        return "H2D"
    if op == "Evict":
        return "D2H"
    return "P2P"


def _find_active_window(events: list[dict], duration_ms: float = 500.0) -> tuple[float, float]:
    """Find a window with the most H2D activity on GPU1 (the PP contention hotspot).

    Prioritizes windows where H2D events on GPU1 co-occur with PP_Recv,
    since that's where the scheduling story is most visible.
    """
    # Score H2D events on GPU1 higher (contention hotspot)
    gpu1_h2d = [e for e in events
                if e["op_type"] in ("Prefetch", "Restore") and e.get("gpu_id") == 1]
    all_h2d = [e for e in events if e["op_type"] in ("Prefetch", "Restore")]
    target = gpu1_h2d if len(gpu1_h2d) >= 3 else all_h2d

    if not target:
        starts = [e["start_ms"] for e in events if "start_ms" in e]
        if not starts:
            return 0, duration_ms
        return min(starts), min(starts) + duration_ms

    sorted_starts = sorted(e["start_ms"] for e in target)
    best_count = 0
    best_start = sorted_starts[0]
    j = 0
    for i, s in enumerate(sorted_starts):
        while j < len(sorted_starts) and sorted_starts[j] < s + duration_ms:
            j += 1
        count = j - i
        if count > best_count:
            best_count = count
            best_start = s

    # Align to start slightly before the first H2D in the window
    return best_start - 10, best_start + duration_ms - 10


def _detect_idle_windows(events: list[dict], t_start_ms: float, t_end_ms: float,
                         target_gpu: int = 1) -> list[tuple[float, float]]:
    """Detect IDLE windows: gaps between consecutive PP_P2P_Recv events on target GPU.

    PP_Recv duration includes NCCL blocking wait. The gap between consecutive
    PP_Recv events is the IDLE window where no PP communication occupies the
    PCIe link — ideal for scheduling H2D transfers.
    """
    pp_recvs = sorted(
        [e for e in events
         if e["op_type"] == "PP_P2P_Recv" and e.get("gpu_id") == target_gpu],
        key=lambda x: x["start_ms"],
    )
    if len(pp_recvs) < 2:
        return []

    idle_windows = []
    for i in range(len(pp_recvs) - 1):
        gap_start = pp_recvs[i]["start_ms"] + pp_recvs[i]["duration_ms"]
        gap_end = pp_recvs[i + 1]["start_ms"]
        gap_ms = gap_end - gap_start
        # IDLE windows: gap between PP_Recv completions
        if gap_ms > 1.0:
            if gap_start < t_end_ms and gap_end > t_start_ms:
                idle_windows.append((
                    max(gap_start, t_start_ms) - t_start_ms,
                    min(gap_end, t_end_ms) - t_start_ms,
                ))
    return idle_windows


def figure_m2(baseline_trace_path: str, sched_trace_path: str | None,
              output_dir: str, window_start: float | None = None,
              window_duration: float = 200.0):
    """M2: PCIe transfer timeline showing IDLE windows and contention."""
    baseline = load_events(baseline_trace_path)
    has_sched = sched_trace_path is not None
    sched = load_events(sched_trace_path) if has_sched else None

    n_panels = 2 if has_sched else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels),
                             squeeze=False)

    panels = [("Baseline (no scheduling)", baseline)]
    if has_sched:
        panels.append(("With PCIe Scheduling", sched))

    for idx, (title, evts) in enumerate(panels):
        ax = axes[idx, 0]

        # Each panel gets its own window (traces may have different absolute times)
        if window_start is not None and idx == 0:
            t_start = window_start
            t_end = window_start + window_duration
        else:
            t_start, t_end = _find_active_window(evts, window_duration)
        print(f"[M2] Panel '{title}' window: {t_start:.1f} - {t_end:.1f} ms")

        # Draw IDLE windows as green bands
        idle_windows = _detect_idle_windows(evts, t_start, t_end)
        print(f"[M2]   IDLE windows: {len(idle_windows)}")
        for iw_start, iw_end in idle_windows:
            ax.axvspan(iw_start, iw_end, alpha=0.08, color="#2ecc71",
                       zorder=0)

        # Draw events
        for e in evts:
            start_ms = e.get("start_ms", e["start_us"] / 1000.0)
            dur = e.get("duration_ms", (e["end_us"] - e["start_us"]) / 1000.0)
            rel_start = start_ms - t_start
            if rel_start + dur < 0 or rel_start > (t_end - t_start):
                continue

            gpu = e.get("gpu_id", 0)
            if gpu > 1:
                continue
            direction = _infer_direction(e)
            track_key = (gpu, direction)
            y = TRACK_ORDER.get(track_key)
            if y is None:
                continue
            color = COLOR_MAP.get(e["op_type"], "#bdc3c7")
            ax.barh(y, dur, left=rel_start, height=0.7, color=color,
                    alpha=0.85, edgecolor="none", zorder=2)

        ax.set_yticks(range(len(TRACK_LABELS)))
        ax.set_yticklabels(TRACK_LABELS, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(0, t_end - t_start)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.15, axis="x")

    axes[-1, 0].set_xlabel("Relative Time (ms)")

    # Legend
    legend_items = ["Prefetch", "Restore", "Evict", "PP_P2P_Recv", "PP_P2P_Send"]
    patches = [mpatches.Patch(color=COLOR_MAP[k], label=k) for k in legend_items
               if k in COLOR_MAP]
    patches.append(mpatches.Patch(color="#2ecc71", alpha=0.15, label="IDLE Window"))
    fig.legend(handles=patches, loc="upper right", fontsize=8, ncol=3,
               bbox_to_anchor=(0.98, 0.98))

    fig.suptitle("PCIe Transfer Timeline with PP Communication Phases",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    out = os.path.join(output_dir, "m2_pcie_timeline.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[M2] Saved: {out}")
    # IDLE window stats already printed per panel above


# ============================================================
# M3: TTFT CDF (multi-group comparison)
# ============================================================
def figure_m3(ttft_groups: dict[str, np.ndarray], output_dir: str):
    """M3: TTFT CDF comparing multiple experimental groups."""
    if not ttft_groups:
        print("M3: No TTFT data provided, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel (a): Full CDF ---
    for group_name in sorted(ttft_groups.keys()):
        data = ttft_groups[group_name]
        if len(data) == 0:
            continue
        sorted_d = np.sort(data)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        color = GROUP_COLORS.get(group_name, "#333333")
        label = GROUP_LABELS.get(group_name, group_name)
        ax1.plot(sorted_d, cdf, label=f"{label} (n={len(data)})",
                 color=color, linewidth=1.8)

    ax1.set_xlabel("TTFT (ms)")
    ax1.set_ylabel("CDF")
    ax1.set_title("(a) TTFT Distribution (all requests)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # --- Panel (b): Bar chart of percentiles ---
    groups_sorted = sorted(ttft_groups.keys())
    x = np.arange(len(groups_sorted))
    width = 0.2

    percentiles = {"P50": 50, "P95": 95, "P99": 99}
    pct_colors = {"P50": "#3498db", "P95": "#e67e22", "P99": "#e74c3c"}

    for i, (pct_name, pct_val) in enumerate(percentiles.items()):
        vals = []
        for g in groups_sorted:
            data = ttft_groups[g]
            vals.append(np.percentile(data, pct_val) if len(data) > 0 else 0)
        bars = ax2.bar(x + (i - 1) * width, vals, width, label=pct_name,
                       color=pct_colors[pct_name], alpha=0.8)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, v,
                         f"{v:.0f}", ha="center", va="bottom", fontsize=6)

    ax2.set_xticks(x)
    labels_for_bar = [GROUP_LABELS.get(g, g).replace(" ", "\n") for g in groups_sorted]
    ax2.set_xticklabels(labels_for_bar, fontsize=7)
    ax2.set_ylabel("TTFT (ms)")
    ax2.set_title("(b) TTFT Percentiles")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Time To First Token Distribution", fontsize=13, y=1.02)
    fig.tight_layout()

    out = os.path.join(output_dir, "m3_ttft_cdf.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[M3] Saved: {out}")

    # Print summary
    print("[M3] TTFT Summary:")
    for g in groups_sorted:
        data = ttft_groups[g]
        if len(data) == 0:
            continue
        print(f"  {g}: n={len(data)}, mean={np.mean(data):.0f}ms, "
              f"P50={np.median(data):.0f}ms, P95={np.percentile(data, 95):.0f}ms, "
              f"P99={np.percentile(data, 99):.0f}ms")


# ============================================================
# M4: Bandwidth Utilization vs. H2D Latency
# ============================================================
def figure_m4(trace_path: str, output_dir: str, slice_ms: float = 50.0):
    """M4: PCIe bandwidth utilization is low, but H2D latency variance is high.

    Shows that the bottleneck is scheduling, not bandwidth.
    Computes time-slice based BW utilization and correlates with H2D latency.

    Uses only real data transfers (H2D, D2H) for BW calculation.
    PP_P2P_Send is ~0.06ms (async NCCL call), so we estimate PP's actual
    PCIe transfer from the PP_P2P_Recv's associated data size divided by
    a reasonable bandwidth (~12 GB/s observed).
    """
    events = load_events(trace_path)

    h2d_events = [e for e in events if e["op_type"] in ("Prefetch", "Restore")]
    d2h_events = [e for e in events if e["op_type"] == "Evict"]
    if not h2d_events:
        print("M4: No H2D events, skipping.")
        return

    # --- Compute global BW utilization using time slices ---
    # Only count actual data movement: H2D (Prefetch/Restore), D2H (Evict)
    # PP P2P uses the same link but PP_Send duration is bogus (0.06ms),
    # so we skip it for BW calculation and rely on overlap classification instead
    data_transfers = h2d_events + d2h_events

    all_start = min(e["start_us"] for e in data_transfers)
    all_end = max(e["end_us"] for e in data_transfers)
    total_dur_s = (all_end - all_start) / 1e6

    total_h2d_bytes = sum(e.get("wire_bytes", e["size_bytes"]) for e in h2d_events)
    total_d2h_bytes = sum(e.get("wire_bytes", e["size_bytes"]) for e in d2h_events)
    total_h2d_active_ms = sum(e["duration_ms"] for e in h2d_events)
    total_d2h_active_ms = sum(e["duration_ms"] for e in d2h_events)

    # PCIe Gen4 x16: ~25.6 GB/s per direction (A100 uses PCIe Gen4)
    PCIE_PEAK_GBPS = 25.6

    # Average BW utilization = total_bytes / (total_time * peak_bw)
    avg_bw_util = ((total_h2d_bytes + total_d2h_bytes) / 1e9) / (total_dur_s * PCIE_PEAK_GBPS) * 100

    # Per-H2D: classify by PP overlap (reuse M1 logic)
    pp_recv_by_gpu = defaultdict(list)
    for e in events:
        if e["op_type"] == "PP_P2P_Recv":
            pp_recv_by_gpu[e["gpu_id"]].append(e)
    for gpu in pp_recv_by_gpu:
        pp_recv_by_gpu[gpu].sort(key=lambda x: x["start_us"])

    overlapped_lats = []
    non_overlapped_lats = []
    all_lats = []
    all_bws = []  # per-transfer achieved bandwidth

    for h in h2d_events:
        h_start, h_end = h["start_us"], h["end_us"]
        h_dur_us = h_end - h_start
        if h_dur_us <= 0:
            continue

        gpu = h.get("gpu_id", 0)
        pp_events = pp_recv_by_gpu.get(gpu, [])
        has_pp = any(
            p["end_us"] > h_start and p["start_us"] < h_end
            for p in pp_events
        )

        lat = h["duration_ms"]
        bw = h.get("bandwidth_gbps", 0)
        all_lats.append(lat)
        all_bws.append(bw)

        if has_pp:
            overlapped_lats.append(lat)
        else:
            non_overlapped_lats.append(lat)

    all_lats = np.array(all_lats)
    all_bws = np.array(all_bws)
    overlapped_lats = np.array(overlapped_lats)
    non_overlapped_lats = np.array(non_overlapped_lats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel (a): Per-transfer achieved BW vs latency ---
    mask_ov = np.zeros(len(all_lats), dtype=bool)
    idx = 0
    for h in h2d_events:
        if h["end_us"] - h["start_us"] <= 0:
            continue
        gpu = h.get("gpu_id", 0)
        pp_events = pp_recv_by_gpu.get(gpu, [])
        has_pp = any(
            p["end_us"] > h["start_us"] and p["start_us"] < h["end_us"]
            for p in pp_events
        )
        mask_ov[idx] = has_pp
        idx += 1

    mask_no_ov = ~mask_ov

    if np.sum(mask_no_ov) > 0:
        ax1.scatter(all_bws[mask_no_ov], all_lats[mask_no_ov],
                    alpha=0.5, s=20, color="#2ecc71", label="No PP overlap",
                    edgecolors="none")
    if np.sum(mask_ov) > 0:
        ax1.scatter(all_bws[mask_ov], all_lats[mask_ov],
                    alpha=0.5, s=20, color="#e74c3c", label="With PP overlap",
                    edgecolors="none")

    ax1.set_xlabel("Achieved Bandwidth per Transfer (GB/s)")
    ax1.set_ylabel("H2D Transfer Latency (ms)")
    ax1.set_title("(a) Per-Transfer BW vs. Latency")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel (b): Key statistics bar chart ---
    categories = ["Overall\nBW Util", "H2D Active\nTime Ratio", "Latency\nCV (no PP)", "Latency\nCV (w/ PP)"]
    values = [
        avg_bw_util,
        total_h2d_active_ms / (total_dur_s * 1000) * 100,  # H2D duty cycle
        (np.std(non_overlapped_lats) / np.mean(non_overlapped_lats) * 100
         if len(non_overlapped_lats) > 0 and np.mean(non_overlapped_lats) > 0 else 0),
        (np.std(overlapped_lats) / np.mean(overlapped_lats) * 100
         if len(overlapped_lats) > 0 and np.mean(overlapped_lats) > 0 else 0),
    ]
    colors_bar = ["#3498db", "#3498db", "#e74c3c", "#e74c3c"]

    x = np.arange(len(categories))
    bars = ax2.bar(x, values, 0.5, color=colors_bar, alpha=0.7)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=8)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("(b) Low Utilization, High Variance")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("PCIe Bandwidth Is Not the Bottleneck — Scheduling Is",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    out = os.path.join(output_dir, "m4_bw_vs_latency.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[M4] Saved: {out}")

    # Print key numbers for paper
    print(f"[M4] Key numbers:")
    print(f"  Experiment duration: {total_dur_s:.1f}s")
    print(f"  Total H2D: {total_h2d_bytes/1e9:.2f} GB in {total_h2d_active_ms:.0f}ms active")
    print(f"  Total D2H: {total_d2h_bytes/1e9:.2f} GB in {total_d2h_active_ms:.0f}ms active")
    print(f"  Average PCIe BW utilization: {avg_bw_util:.2f}%")
    print(f"  H2D duty cycle: {total_h2d_active_ms / (total_dur_s * 1000) * 100:.2f}%")
    print(f"  H2D latency (all): mean={np.mean(all_lats):.2f}ms, CV={np.std(all_lats)/np.mean(all_lats)*100:.1f}%")
    if len(non_overlapped_lats) > 0:
        print(f"  H2D latency (no PP): mean={np.mean(non_overlapped_lats):.2f}ms, "
              f"CV={np.std(non_overlapped_lats)/np.mean(non_overlapped_lats)*100:.1f}%")
    if len(overlapped_lats) > 0:
        print(f"  H2D latency (w/ PP): mean={np.mean(overlapped_lats):.2f}ms, "
              f"CV={np.std(overlapped_lats)/np.mean(overlapped_lats)*100:.1f}%")


# ============================================================
# Auto-detect files in experiment directory
# ============================================================
def auto_detect_files(exp_dir: str) -> dict:
    """Auto-detect PCIe trace and TTFT files in an experiment directory."""
    d = Path(exp_dir)
    result = {}

    # PCIe traces
    for pattern, key in [
        ("pcie_events_g0_*", "trace_g0"),
        ("pcie_events_g1_*", "trace_g1"),
        ("pcie_events_g2_*", "trace_g2"),
        ("pcie_events_g3_*", "trace_g3"),
    ]:
        matches = list(d.glob(pattern))
        if matches:
            result[key] = str(matches[0])

    # TTFT files
    for pattern, key in [
        ("prefetch_g0_*", "ttft_g0"),
        ("prefetch_g1_*", "ttft_g1"),
        ("prefetch_g2_*", "ttft_g2"),
        ("prefetch_g3_*", "ttft_g3"),
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
        description="Generate motivation section figures for PCIe scheduling paper")
    parser.add_argument("--exp-dir", help="Experiment result directory (auto-detect files)")
    parser.add_argument("--baseline-trace", help="PCIe trace JSON for baseline (G1)")
    parser.add_argument("--sched-trace", help="PCIe trace JSON for scheduled (G3)")
    parser.add_argument("--ttft-files", nargs="*",
                        help="'GroupName:path.jsonl' pairs for TTFT analysis")
    parser.add_argument("--figures", nargs="*", default=["m1", "m2", "m3", "m4"],
                        help="Which figures to generate (default: all)")
    parser.add_argument("--output", default="figures/motivation/",
                        help="Output directory")
    parser.add_argument("--window-start", type=float, default=None,
                        help="M2 timeline window start (ms, absolute)")
    parser.add_argument("--window-duration", type=float, default=500.0,
                        help="M2 timeline window duration (ms, default=500)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    figures = set(f.lower() for f in args.figures)

    # Auto-detect files from exp-dir
    files = {}
    if args.exp_dir:
        files = auto_detect_files(args.exp_dir)
        print(f"Auto-detected files in {args.exp_dir}:")
        for k, v in sorted(files.items()):
            print(f"  {k}: {Path(v).name}")

    # Resolve trace paths
    baseline_trace = args.baseline_trace or files.get("trace_g1")
    sched_trace = args.sched_trace or files.get("trace_g3")

    # Resolve TTFT paths
    ttft_groups = {}
    if args.ttft_files:
        for item in args.ttft_files:
            if ":" not in item:
                continue
            label, path = item.split(":", 1)
            data = load_ttft(path)
            if len(data) > 0:
                ttft_groups[label] = data
    else:
        for group_key, file_key in [("G0", "ttft_g0"), ("G1", "ttft_g1"),
                                     ("G2", "ttft_g2"), ("G3", "ttft_g3")]:
            if file_key in files:
                data = load_ttft(files[file_key])
                if len(data) > 0:
                    ttft_groups[group_key] = data

    # Generate figures
    if "m1" in figures:
        # M1 needs traces from multiple groups for comparison
        m1_traces = {}
        for gkey, fkey in [("G0", "trace_g0"), ("G1", "trace_g1"),
                           ("G2", "trace_g2"), ("G3", "trace_g3")]:
            if fkey in files:
                m1_traces[gkey] = files[fkey]
        if m1_traces:
            print("\n" + "=" * 60)
            print("Generating M1: H2D Traffic Characterization")
            print("=" * 60)
            figure_m1(m1_traces, args.output)
        else:
            print("M1: No trace files found, skipping.")

    if "m2" in figures:
        if baseline_trace:
            print("\n" + "=" * 60)
            print("Generating M2: PCIe Timeline")
            print("=" * 60)
            figure_m2(baseline_trace, sched_trace, args.output,
                      window_start=args.window_start,
                      window_duration=args.window_duration)
        else:
            print("M2: No trace files found, skipping.")

    if "m3" in figures:
        if ttft_groups:
            print("\n" + "=" * 60)
            print("Generating M3: TTFT CDF")
            print("=" * 60)
            figure_m3(ttft_groups, args.output)
        else:
            print("M3: No TTFT data found, skipping.")

    if "m4" in figures:
        trace_for_m4 = baseline_trace
        if trace_for_m4:
            print("\n" + "=" * 60)
            print("Generating M4: BW Utilization vs. Latency")
            print("=" * 60)
            figure_m4(trace_for_m4, args.output)
        else:
            print("M4: No trace found, skipping.")


if __name__ == "__main__":
    main()
