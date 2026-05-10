#!/usr/bin/env python3
"""
极端场景压力实验 — 出版质量图表生成（v2）

叙事重心：预取失败是否比"没有预取"更差？

  图 A: S1 速率限制削减调度器负载（柱状图）
  图 B: S2 TPOT 对比 §3.3.3 正常运行曲线（折线 + 标记）
  图 C: GPU 每卡功耗对比

用法:
  python plot_stress_figures.py \
    --root /path/to/stress/results \
    --output /path/to/stress/results/figs
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 全局样式
# ============================================================
plt.rcParams.update({
    "font.size": 14,
    "font.family": ["Times New Roman", "Songti SC"],
    "axes.unicode_minus": False,
    "axes.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "grid.linewidth": 0.5,
    "lines.linewidth": 2.0,
})

C_VLLM = "#1f77b4"        # 蓝 — vLLM 基线
C_PREFETCH = "#2ca02c"     # 绿 — LLM-Prefetch
C_STRESS = "#E74C3C"       # 红 — 压力测试点
C_PROTECTED = "#2980B9"
C_UNPROTECTED = "#7F8C8D"
C_RATELIMIT = "#27AE60"

# §3.3.3 eval_combined 数据 (Qwen2.5-72B, TP=2)
EVAL_QPS = [0.2, 0.4, 0.6, 0.8, 1.0]
EVAL_VLLM_TPOT = [34.67, 39.65, 50.69, 61.46, 71.1]
EVAL_PF_TPOT = [31.15, 37.46, 48.68, 57.58, 68.2]


# ============================================================
# 数据工具
# ============================================================
def load_aggregate(root: Path) -> list[dict]:
    path = root / "aggregate.csv"
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_reps(rows: list[dict], prefix: str) -> list[dict]:
    return [r for r in rows if r["unit"].startswith(prefix)]


def mean_std(values: list[float]) -> tuple[float, float]:
    m = np.mean(values)
    s = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return float(m), float(s)


# ============================================================
# 图 A: S1 速率限制效果
# ============================================================
def figure_a(rows: list[dict], output: str):
    configs = [
        ("无防御", "unprotected_s1_qps200", C_UNPROTECTED),
        ("预取配额+TTL", "protected_s1_qps200", C_PROTECTED),
        ("预取配额+TTL\n+速率限制", "ratelimit_s1_qps200", C_RATELIMIT),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)

    xs = np.arange(len(configs))
    width = 0.55
    means, stds = [], []

    for label, prefix, color in configs:
        reps = get_reps(rows, prefix)
        vals = [float(r["prom_no_hits"]) for r in reps]
        m, s = mean_std(vals)
        means.append(m)
        stds.append(s)

    ax.bar(xs, means, width, yerr=stds, capsize=5,
           color=[c for _, _, c in configs],
           edgecolor="white", linewidth=0.5,
           error_kw={"linewidth": 1.2})

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 300, f"{m:,.0f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([l for l, _, _ in configs], fontsize=13)
    ax.set_ylabel("到达调度器的请求数", fontsize=14)
    ax.set_ylim(0, max(means) * 1.25)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.grid(True, alpha=0.2, axis="y")

    out = os.path.join(output, "fig_s1_ratelimit.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[图 A] S1 速率限制效果 -> {out}")


# ============================================================
# 图 B: S2 TPOT — 与 §3.3.3 正常运行对比
# ============================================================
def figure_b(rows: list[dict], output: str):
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)

    # §3.3.3 性能曲线
    ax.plot(EVAL_QPS, EVAL_VLLM_TPOT,
            color=C_VLLM, marker="^", linestyle="-.", markersize=8,
            label="vLLM 基线（§3.3.3）", zorder=3)
    ax.plot(EVAL_QPS, EVAL_PF_TPOT,
            color=C_PREFETCH, marker="o", linestyle="-", markersize=8,
            label="LLM-Prefetch 正常（§3.3.3）", zorder=3)

    # S2 stress test TPOT
    s2_reps = get_reps(rows, "protected_s2_burst5")
    s2_tpot = mean_std([float(r["real_tpot_mean_ms"]) for r in s2_reps])[0]
    s2_qps = 0.59  # real_sent / duration

    ax.plot(s2_qps, s2_tpot, marker="*", color=C_STRESS,
            markersize=16, zorder=5, markeredgecolor="white", markeredgewidth=0.8,
            label=f"S2 误触压力测试 ({s2_tpot:.1f} ms)")

    # 标注有效负载更高
    ax.annotate("有效负载含\nburst 预取开销",
                xy=(s2_qps, s2_tpot),
                xytext=(s2_qps + 0.18, s2_tpot + 3),
                fontsize=11, color=C_STRESS,
                arrowprops=dict(arrowstyle="->", color=C_STRESS, lw=1.2),
                zorder=6)

    ax.set_xlabel("负载 (QPS)", fontsize=14)
    ax.set_ylabel("平均 TPOT (ms)", fontsize=14)
    ax.set_xticks(EVAL_QPS)
    ax.set_xlim(0.1, 1.1)
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9, edgecolor="gray")
    ax.grid(True, alpha=0.2)

    out = os.path.join(output, "fig_s2_tpot_envelope.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[图 B] S2 TPOT 对比 -> {out}")


# ============================================================
# 图 C: GPU 每卡功耗对比
# ============================================================
def figure_c(rows: list[dict], output: str):
    configs = [
        ("S1\n无防御", "unprotected_s1_qps200", C_UNPROTECTED),
        ("S1\n预取配额+TTL", "protected_s1_qps200", C_PROTECTED),
        ("S1\n全部防御", "ratelimit_s1_qps200", C_RATELIMIT),
        ("S2\n无防御", "unprotected_s2_burst5", C_UNPROTECTED),
        ("S2\n预取配额+TTL", "protected_s2_burst5", C_PROTECTED),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    xs = np.arange(len(configs))
    width = 0.55
    means, stds, colors = [], [], []

    for label, prefix, color in configs:
        reps = get_reps(rows, prefix)
        vals = [float(r["gpu_mean_w"]) / 2.0 for r in reps]
        m, s = mean_std(vals)
        means.append(m)
        stds.append(s)
        colors.append(color)

    ax.bar(xs, means, width, yerr=stds, capsize=4,
           color=colors, edgecolor="white", linewidth=0.5,
           error_kw={"linewidth": 1.2})

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 4, f"{m:.0f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.axhline(y=85, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(len(configs) - 0.5, 89, "GPU 空闲功耗", ha="right", va="bottom",
            fontsize=12, color="gray")

    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(1, max(means) * 1.10, "S1: 泛洪攻击\n（无真实推理）",
            ha="center", va="bottom", fontsize=12, color="#555")
    ax.text(3.5, max(means) * 1.10, "S2: 误触 Burst\n（含真实推理）",
            ha="center", va="bottom", fontsize=12, color="#555")

    ax.set_xticks(xs)
    ax.set_xticklabels([l for l, _, _ in configs], fontsize=12)
    ax.set_ylabel("每卡平均功耗 (W)", fontsize=14)
    ax.set_ylim(0, max(means) * 1.25)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.grid(True, alpha=0.2, axis="y")

    out = os.path.join(output, "fig_gpu_power.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[图 C] GPU 功耗对比 -> {out}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    output = args.output or str(root / "figs")
    os.makedirs(output, exist_ok=True)

    rows = load_aggregate(root)
    print(f"加载 {len(rows)} 组实验数据")

    figure_a(rows, output)
    figure_b(rows, output)
    figure_c(rows, output)
    print(f"\n所有图表已保存至 {output}/")


if __name__ == "__main__":
    main()
