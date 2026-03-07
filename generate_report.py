#!/usr/bin/env python3
"""
Prefetch A/B 实验报告生成器

读取 baseline.jsonl 和 prefetch.jsonl，生成 HTML 可视化报告。
包含：TTFT CDF、cached_tokens 分析、配对对比、汇总表。

用法:
  python generate_report.py --baseline results/baseline.jsonl --prefetch results/prefetch.jsonl --output results/report.html
"""

import argparse
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def compute_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cdf = np.arange(1, n + 1) / n
    return sorted_vals, cdf


def main():
    parser = argparse.ArgumentParser(description="Generate Prefetch A/B Report")
    parser.add_argument("--baseline", required=True, help="Baseline JSONL 路径")
    parser.add_argument("--prefetch", required=True, help="Prefetch JSONL 路径")
    parser.add_argument("--output", required=True, help="输出 HTML 路径")
    args = parser.parse_args()

    df_b = load_jsonl(args.baseline)
    df_p = load_jsonl(args.prefetch)

    df_b = df_b[df_b["success"]].copy()
    df_p = df_p[df_p["success"]].copy()

    if df_b.empty or df_p.empty:
        print("警告: 无成功请求，报告可能不完整")

    figs = []

    # 1. TTFT CDF
    fig_ttft = go.Figure()
    if not df_b.empty:
        x_b, cdf_b = compute_cdf(df_b["ttft_ms"].values)
        fig_ttft.add_trace(go.Scatter(x=x_b, y=cdf_b, mode="lines", name="Baseline", line=dict(width=2)))
    if not df_p.empty:
        x_p, cdf_p = compute_cdf(df_p["ttft_ms"].values)
        fig_ttft.add_trace(go.Scatter(x=x_p, y=cdf_p, mode="lines", name="Prefetch", line=dict(width=2)))
    fig_ttft.update_layout(
        title="TTFT CDF (Time To First Token)",
        xaxis_title="TTFT (ms)",
        yaxis_title="CDF",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    figs.append(("TTFT CDF", fig_ttft))

    # 2. TTFT Box Plot
    fig_box = go.Figure()
    if not df_b.empty:
        fig_box.add_trace(go.Box(y=df_b["ttft_ms"], name="Baseline", boxpoints="outliers"))
    if not df_p.empty:
        fig_box.add_trace(go.Box(y=df_p["ttft_ms"], name="Prefetch", boxpoints="outliers"))
    fig_box.update_layout(
        title="TTFT Distribution (Box Plot)",
        yaxis_title="TTFT (ms)",
    )
    figs.append(("TTFT Box", fig_box))

    # 3. 分位数表
    percentiles = [50, 90, 95, 99]
    p_b = {p: np.percentile(df_b["ttft_ms"], p) if not df_b.empty else 0 for p in percentiles}
    p_p = {p: np.percentile(df_p["ttft_ms"], p) if not df_p.empty else 0 for p in percentiles}
    pct_improve = {}
    for p in percentiles:
        if p_b[p] > 0:
            pct_improve[p] = (1 - p_p[p] / p_b[p]) * 100
        else:
            pct_improve[p] = 0

    table_data = [
        ["P50", f"{p_b[50]:.2f}", f"{p_p[50]:.2f}", f"{pct_improve[50]:.1f}%"],
        ["P90", f"{p_b[90]:.2f}", f"{p_p[90]:.2f}", f"{pct_improve[90]:.1f}%"],
        ["P95", f"{p_b[95]:.2f}", f"{p_p[95]:.2f}", f"{pct_improve[95]:.1f}%"],
        ["P99", f"{p_b[99]:.2f}", f"{p_p[99]:.2f}", f"{pct_improve[99]:.1f}%"],
    ]
    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["", "Baseline (ms)", "Prefetch (ms)", "TTFT 提升"],
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=[[r[0] for r in table_data], [r[1] for r in table_data], [r[2] for r in table_data], [r[3] for r in table_data]],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    fig_table.update_layout(title="TTFT 分位数对比")
    figs.append(("TTFT 分位数", fig_table))

    # 4. cached_tokens 命中分析
    if "cached_tokens" in df_p.columns:
        df_p_cached = df_p[df_p["cached_tokens"].notna()].copy()
        hit_rate = (df_p_cached["cached_tokens"] > 0).mean() * 100 if not df_p_cached.empty else 0
        if not df_p_cached.empty:
            df_p_cached["cached_ratio"] = df_p_cached[["cached_tokens", "prompt_tokens"]].apply(
                lambda r: r["cached_tokens"] / r["prompt_tokens"] if r["prompt_tokens"] > 0 else 0, axis=1
            )
    else:
        df_p_cached = pd.DataFrame()
        hit_rate = 0

    fig_cached = go.Figure()
    if not df_p_cached.empty and "cached_ratio" in df_p_cached.columns and (df_p_cached["cached_tokens"] > 0).any():
        fig_cached.add_trace(go.Histogram(x=df_p_cached["cached_ratio"], name="cached/prompt", nbinsx=30))
    fig_cached.update_layout(
        title=f"cached_tokens / prompt_tokens 分布 (Prefetch 组, 命中率={hit_rate:.1f}%)",
        xaxis_title="cached_tokens / prompt_tokens",
        yaxis_title="Count",
    )
    figs.append(("cached_tokens 分布", fig_cached))

    # 5. 配对对比
    merged = pd.merge(
        df_b[["chat_id", "turn", "ttft_ms"]],
        df_p[["chat_id", "turn", "ttft_ms"]],
        on=["chat_id", "turn"],
        suffixes=("_baseline", "_prefetch"),
    )
    if not merged.empty:
        merged["ttft_diff_ms"] = merged["ttft_ms_baseline"] - merged["ttft_ms_prefetch"]
        fig_pair = make_subplots(rows=1, cols=2, subplot_titles=("TTFT 差值分布", "配对散点"))
        fig_pair.add_trace(go.Histogram(x=merged["ttft_diff_ms"], nbinsx=30), row=1, col=1)
        fig_pair.add_trace(
            go.Scatter(x=merged["ttft_ms_baseline"], y=merged["ttft_ms_prefetch"], mode="markers", name="配对"),
            row=1,
            col=2,
        )
        mx = max(merged["ttft_ms_baseline"].max(), merged["ttft_ms_prefetch"].max(), 1)
        fig_pair.add_trace(
            go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="y=x", line=dict(dash="dash")),
            row=1,
            col=2,
        )
        fig_pair.update_xaxes(title_text="TTFT 差值 (ms)", row=1, col=1)
        fig_pair.update_xaxes(title_text="Baseline TTFT (ms)", row=1, col=2)
        fig_pair.update_yaxes(title_text="Prefetch TTFT (ms)", row=1, col=2)
        fig_pair.update_layout(title=f"配对对比 (n={len(merged)})")
        mean_diff = merged["ttft_diff_ms"].mean()
        print(f"配对样本数: {len(merged)}, 平均 TTFT 差值: {mean_diff:.2f} ms (baseline - prefetch)")
    else:
        fig_pair = go.Figure()
        fig_pair.add_annotation(text="无配对样本", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig_pair.update_layout(title="配对对比")
    figs.append(("配对对比", fig_pair))

    # 6. 汇总 Summary
    mean_b = df_b["ttft_ms"].mean() if not df_b.empty else 0
    mean_p = df_p["ttft_ms"].mean() if not df_p.empty else 0
    mean_improve = (1 - mean_p / mean_b) * 100 if mean_b > 0 else 0

    summary_html = f"""
    <h2>汇总 Summary</h2>
    <ul>
    <li>Baseline 样本数: {len(df_b)}</li>
    <li>Prefetch 样本数: {len(df_p)}</li>
    <li>平均 TTFT 降低: {mean_improve:.1f}%</li>
    <li>Prefetch cached_tokens 命中率: {hit_rate:.1f}%</li>
    </ul>
    """

    html_parts = [f"<html><head><meta charset='utf-8'><title>Prefetch A/B Report</title></head><body>"]
    html_parts.append("<h1>Prefetch KV Cache A/B 评估报告</h1>")
    html_parts.append(summary_html)

    for name, fig in figs:
        html_parts.append(f"<h2>{name}</h2>")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    html_parts.append("</body></html>")
    html = "\n".join(html_parts)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"报告已生成: {args.output}")


if __name__ == "__main__":
    main()
