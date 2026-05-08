#!/usr/bin/env python3
"""Aggregate stress-experiment runs into a single CSV + summary plots.

Layout expected under ``--root``:
  <root>/<unit>/{summary.json, requests.jsonl, power.csv, perf_pkg.txt, pcie_events.json}

For each unit we extract:

  * scenario / config (from summary.json)
  * GPU power: mean / p95 / cumulative energy (J) across all GPUs
  * CPU energy (J) from perf_pkg.txt
  * PCIe wasted-I/O bytes: total bytes of Prefetch-tagged H2D events whose
    target block was never touched (proxy: sum of bytes for events whose
    associated request landed in the no-hit / expired buckets, or simply
    bytes(Prefetch_total) - bytes(Prefetch_kept) if individual mapping is
    not available)
  * Prometheus deltas: gpu_hits / cpu_hits / no_hits / deferred / expired
  * Real-request latency stats (S2/S3 only)

Outputs:
  * <root>/aggregate.csv          one row per unit
  * <root>/figs/power_curve.pdf   per-scenario stack
  * <root>/figs/wasted_io.pdf
  * <root>/figs/ttft_cdf.pdf      S3 only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_units(root: Path) -> list[dict]:
    units = []
    for unit_dir in sorted(root.iterdir()):
        if not unit_dir.is_dir():
            continue
        summary_path = unit_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        units.append({"dir": unit_dir, "summary": summary})
    return units


def gpu_power_stats(unit_dir: Path) -> dict:
    """Aggregate per-GPU power.draw across the sampling window.

    Returns mean_w / p95_w / energy_j (sum across GPUs * window).
    """
    csv_path = unit_dir / "power.csv"
    if not csv_path.exists():
        return {"mean_w": None, "p95_w": None, "energy_j": None}
    per_gpu: dict[str, list[float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                idx = row[1].strip()
                pw = float(row[2])
            except ValueError:
                continue
            per_gpu.setdefault(idx, []).append(pw)
    if not per_gpu:
        return {"mean_w": None, "p95_w": None, "energy_j": None}
    total_mean = 0.0
    p95s: list[float] = []
    energy_j = 0.0
    sample_period_s = 0.1
    for samples in per_gpu.values():
        total_mean += statistics.fmean(samples)
        ordered = sorted(samples)
        if len(ordered) >= 20:
            p95s.append(ordered[int(0.95 * len(ordered)) - 1])
        else:
            p95s.append(max(ordered))
        energy_j += sum(samples) * sample_period_s
    return {
        "mean_w": total_mean,
        "p95_w": sum(p95s),
        "energy_j": energy_j,
    }


def cpu_energy_j(unit_dir: Path) -> float | None:
    txt = unit_dir / "perf_pkg.txt"
    if not txt.exists():
        return None
    pat = re.compile(r"([\d,\.]+)\s+Joules\s+power/energy-pkg/")
    for line in txt.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
    return None


def pcie_stats(unit_dir: Path) -> dict:
    """Aggregate Prefetch H2D events for the run.

    In addition to total bytes / count, derive PCIe wake-up indicators
    from the inter-event gap distribution:

      - ``gap_p50_us`` / ``gap_p95_us`` — median/p95 inter-event gap.
      - ``short_gap_ratio`` — fraction of events whose gap to the
        previous Prefetch is < 100 us. ASPM L0s/L1 wake latencies are in
        the few-to-tens-of-microseconds range, so a high short-gap ratio
        means the link rarely had a chance to enter a low-power state.
      - ``avg_event_bw_mbps`` — average link bandwidth used by Prefetch
        traffic over the wall-clock window covered by the events.

    Wall-clock is approximated by the span between the first and last
    event's ``start_us``. For very low event counts these stats degrade
    gracefully.
    """
    path = unit_dir / "pcie_events.json"
    out = {
        "prefetch_bytes": 0,
        "prefetch_count": 0,
        "gap_p50_us": None,
        "gap_p95_us": None,
        "short_gap_ratio": None,
        "avg_event_bw_mbps": None,
    }
    if not path.exists():
        return out
    try:
        events = json.loads(path.read_text())
    except Exception:
        return out

    pf = [ev for ev in events if ev.get("op_type") == "Prefetch"]
    if not pf:
        return out

    pf.sort(key=lambda e: float(e.get("start_us", 0)))
    bytes_sum = sum(
        int(ev.get("wire_bytes") or ev.get("size_bytes") or 0) for ev in pf
    )
    out["prefetch_bytes"] = bytes_sum
    out["prefetch_count"] = len(pf)

    if len(pf) >= 2:
        gaps_us: list[float] = []
        for i in range(1, len(pf)):
            prev_end = float(pf[i - 1].get("end_us", pf[i - 1].get("start_us", 0)))
            cur_start = float(pf[i].get("start_us", 0))
            gap = cur_start - prev_end
            if gap >= 0:
                gaps_us.append(gap)
        if gaps_us:
            gaps_sorted = sorted(gaps_us)
            out["gap_p50_us"] = gaps_sorted[len(gaps_sorted) // 2]
            p95_idx = max(0, int(0.95 * len(gaps_sorted)) - 1)
            out["gap_p95_us"] = gaps_sorted[p95_idx]
            short = sum(1 for g in gaps_sorted if g < 100.0)
            out["short_gap_ratio"] = short / len(gaps_sorted)

        span_us = float(pf[-1].get("start_us", 0)) - float(pf[0].get("start_us", 0))
        if span_us > 0:
            out["avg_event_bw_mbps"] = (bytes_sum / (1024 * 1024)) / (span_us / 1e6)

    return out


def prom_deltas(samples: list[dict]) -> dict:
    """Convert Prometheus counter samples (cumulative) into deltas over the window."""
    if not samples:
        return {k: 0 for k in (
            "gpu_hits", "cpu_hits", "no_hits", "deferred", "expired"
        )}
    keys = {
        "gpu_hits": "vllm:prefetch_gpu_hits_total",
        "cpu_hits": "vllm:prefetch_cpu_hits_total",
        "no_hits": "vllm:prefetch_no_hits_total",
        "deferred": "vllm:prefetch_deferred_total",
        "expired": "vllm:prefetch_expired_total",
    }
    out: dict[str, float] = {}
    for short, full in keys.items():
        values = [s.get(full, 0.0) for s in samples]
        if not values:
            out[short] = 0.0
            continue
        out[short] = max(values) - min(values)
    return out


def aggregate(units: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for unit in units:
        summary = unit["summary"]
        unit_dir: Path = unit["dir"]
        gpu = gpu_power_stats(unit_dir)
        cpu_j = cpu_energy_j(unit_dir)
        pcie = pcie_stats(unit_dir)
        deltas = prom_deltas(summary.get("prometheus_samples", []))

        attempts = (
            deltas["gpu_hits"]
            + deltas["cpu_hits"]
            + deltas["no_hits"]
            + deltas["deferred"]
            + deltas["expired"]
        )
        # Two complementary "wasted" views:
        #   wasted_attempt_ratio: of all prefetch attempts the engine saw,
        #     what fraction ended up benefiting nobody (no_hits = nothing
        #     to load; expired = loaded then never consumed).
        #   wasted_byte_estimate: prefetch_expired blocks correspond to
        #     bytes that were transferred over PCIe and then reclaimed
        #     without being touched. We don't know exact bytes per block
        #     without server-side accounting, so we report the count and
        #     leave the byte-level estimate for the analysis writeup.
        wasted_attempts = deltas["no_hits"] + deltas["expired"]
        wasted_attempt_ratio = (
            wasted_attempts / attempts if attempts > 0 else 0.0
        )

        latency = summary.get("real_latency_ms", {}) or {}
        rows.append(
            {
                "unit": unit_dir.name,
                "scenario": summary.get("scenario", ""),
                "duration_sec": summary.get("duration_sec", 0),
                "prefetch_qps": (summary.get("config") or {}).get("prefetch_qps"),
                "qps": (summary.get("config") or {}).get("qps"),
                "burst_size": (summary.get("config") or {}).get("burst_size"),
                "mix_ratio": (summary.get("config") or {}).get("mix_ratio"),
                "gpu_mean_w": gpu["mean_w"],
                "gpu_p95_w": gpu["p95_w"],
                "gpu_energy_j": gpu["energy_j"],
                "cpu_energy_j": cpu_j,
                "pcie_prefetch_bytes": pcie["prefetch_bytes"],
                "pcie_prefetch_events": pcie["prefetch_count"],
                "pcie_gap_p50_us": pcie["gap_p50_us"],
                "pcie_gap_p95_us": pcie["gap_p95_us"],
                "pcie_short_gap_ratio": pcie["short_gap_ratio"],
                "pcie_avg_event_bw_mbps": pcie["avg_event_bw_mbps"],
                "prom_gpu_hits": deltas["gpu_hits"],
                "prom_cpu_hits": deltas["cpu_hits"],
                "prom_no_hits": deltas["no_hits"],
                "prom_deferred": deltas["deferred"],
                "prom_expired": deltas["expired"],
                "wasted_attempts": wasted_attempts,
                "wasted_attempt_ratio": wasted_attempt_ratio,
                "real_ttft_mean_ms": latency.get("ttft_mean"),
                "real_ttft_p95_ms": latency.get("ttft_p95"),
                "real_tpot_mean_ms": latency.get("tpot_mean"),
            }
        )
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_power(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    by_unit = sorted(rows, key=lambda r: r["unit"])
    xs = [r["unit"] for r in by_unit]
    ys_mean = [r["gpu_mean_w"] or 0 for r in by_unit]
    ys_p95 = [r["gpu_p95_w"] or 0 for r in by_unit]
    ax.bar(range(len(xs)), ys_mean, label="mean")
    ax.bar(range(len(xs)), ys_p95, alpha=0.4, label="p95")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("GPU power (W, summed across cards)")
    ax.set_title("Stress: GPU power per unit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_wasted_io(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [r["unit"] for r in rows]
    bytes_per_unit = [r["pcie_prefetch_bytes"] / (1 << 30) for r in rows]
    ax.bar(range(len(xs)), bytes_per_unit, color="#888")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Prefetch H2D bytes (GiB, total during window)")
    ax.set_title("Stress: Prefetch PCIe traffic per unit")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_pcie_gap_cdf(units: list[dict], out: Path) -> None:
    """CDF of inter-Prefetch-event gaps per unit.

    Useful for the 'PCIe 唤醒' argument: a curve hugging the y-axis (most
    gaps very short) means the link basically never sleeps; a curve
    shifted to the right means transfers are spaced enough that ASPM
    can take effect between them.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = 0
    for u in units:
        unit_dir = u["dir"]
        path = unit_dir / "pcie_events.json"
        if not path.exists():
            continue
        try:
            events = json.loads(path.read_text())
        except Exception:
            continue
        pf = [ev for ev in events if ev.get("op_type") == "Prefetch"]
        if len(pf) < 2:
            continue
        pf.sort(key=lambda e: float(e.get("start_us", 0)))
        gaps = []
        for i in range(1, len(pf)):
            prev_end = float(
                pf[i - 1].get("end_us", pf[i - 1].get("start_us", 0))
            )
            cur_start = float(pf[i].get("start_us", 0))
            gap = cur_start - prev_end
            if gap >= 0:
                gaps.append(gap)
        if not gaps:
            continue
        gaps.sort()
        ys = [(i + 1) / len(gaps) for i in range(len(gaps))]
        ax.plot(gaps, ys, label=unit_dir.name)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("Inter-Prefetch gap (μs, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title("Stress: PCIe wake-up gap distribution")
    ax.axvline(100.0, color="grey", linestyle="--", linewidth=0.8,
               label="ASPM threshold (100 μs)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_ttft_cdf(units: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = 0
    for u in units:
        unit_dir = u["dir"]
        jsonl = unit_dir / "requests.jsonl"
        if not jsonl.exists():
            continue
        ttfts: list[float] = []
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") != "real":
                    continue
                if rec.get("success") is False:
                    continue
                v = rec.get("ttft_ms")
                if v is not None:
                    ttfts.append(v)
        if not ttfts:
            continue
        ttfts.sort()
        ys = [(i + 1) / len(ttfts) for i in range(len(ttfts))]
        ax.plot(ttfts, ys, label=unit_dir.name)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("Real-request TTFT (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Stress: real-request TTFT CDF")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenario", choices=["s1", "s2", "s3"], default=None)
    args = ap.parse_args()

    root = Path(args.root)
    units = load_units(root)
    if args.scenario:
        units = [u for u in units if u["summary"].get("scenario") == args.scenario]
    rows = aggregate(units)

    write_csv(rows, root / "aggregate.csv")
    figs = root / "figs"
    figs.mkdir(exist_ok=True)
    plot_power(rows, figs / "power_curve.pdf")
    plot_wasted_io(rows, figs / "wasted_io.pdf")
    plot_pcie_gap_cdf(units, figs / "pcie_gap_cdf.pdf")
    plot_ttft_cdf(units, figs / "ttft_cdf.pdf")
    print(f"wrote {len(rows)} rows -> {root / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
