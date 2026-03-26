#!/usr/bin/env python3
"""
Prefetch A/B 实验报告生成器

读取 baseline.jsonl 和 prefetch.jsonl，生成 Markdown 报告。
包含：TTFT/TPOT 统计表、cached_tokens 分析、配对对比、详细 vLLM 与测试配置。

用法:
  python generate_report.py --baseline results/baseline.jsonl --prefetch results/prefetch.jsonl --output results/report.md
  # 三联（普通 vLLM / +Prefetch / +Prefetch+PCIe）:
  python generate_report.py --plain plain.jsonl --baseline prefetch.jsonl --prefetch prefetch_pcie.jsonl -o report.md
"""

import argparse
import json
import os
import numpy as np
import pandas as pd


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _format_config_section(title: str, config_str: str) -> str:
    """将配置字符串格式化为 Markdown 列表。支持逗号或换行分隔的 key=value。"""
    if not config_str or config_str.strip() == "未指定":
        return ""
    lines = []
    for part in config_str.replace(",", "\n").split("\n"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            lines.append(f"- **{k.strip()}**: {v.strip()}")
        elif part:
            lines.append(f"- {part}")
    if not lines:
        return ""
    return f"### {title}\n\n" + "\n".join(lines) + "\n\n"


def main():
    parser = argparse.ArgumentParser(description="Generate Prefetch A/B Report (Markdown)")
    parser.add_argument(
        "--plain",
        type=str,
        default="",
        help="普通 vLLM JSONL（无客户端 prefetch）；若路径存在则生成三联对比",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="+Prefetch（无 PCIe 调度）JSONL；兼容旧用法时也可作单一 baseline",
    )
    parser.add_argument("--prefetch", required=True, help="+Prefetch+PCIe 调度 JSONL")
    parser.add_argument("--output", required=True, help="输出 Markdown 路径")
    parser.add_argument("--config", type=str, default="", help="配置摘要")
    parser.add_argument("--config-file", type=str, default="", help="从文件读取完整配置（key=value 格式，用于填充配置详情）")
    parser.add_argument("--vllm-config", type=str, default="", help="vLLM 配置详情")
    parser.add_argument("--test-config", type=str, default="", help="测试配置详情")
    args = parser.parse_args()

    # 若指定 --config-file，从文件读取配置（覆盖空的 --config）
    if args.config_file and (not args.config or args.config.strip() == ""):
        try:
            with open(args.config_file, "r", encoding="utf-8") as f:
                lines = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        lines.append(line)
                args.config = "\n".join(lines)
        except OSError:
            pass

    df_b = load_jsonl(args.baseline)
    df_p = load_jsonl(args.prefetch)

    df_b = df_b[df_b["success"]].copy()
    df_p = df_p[df_p["success"]].copy()

    df_plain = pd.DataFrame()
    three_way = False
    if args.plain.strip() and os.path.isfile(args.plain):
        df_plain = load_jsonl(args.plain)
        if not df_plain.empty and "success" in df_plain.columns:
            df_plain = df_plain[df_plain["success"]].copy()
        three_way = not df_plain.empty

    if df_b.empty or df_p.empty:
        print("警告: 无成功请求，报告可能不完整")

    config_str = args.config or "未指定"

    # 分位数
    percentiles = [50, 90, 95, 99]
    p_plain = {
        p: np.percentile(df_plain["ttft_ms"], p) if not df_plain.empty else 0 for p in percentiles
    }
    p_b = {p: np.percentile(df_b["ttft_ms"], p) if not df_b.empty else 0 for p in percentiles}
    p_p = {p: np.percentile(df_p["ttft_ms"], p) if not df_p.empty else 0 for p in percentiles}
    pct_improve = {}
    for p in percentiles:
        if p_b[p] > 0:
            pct_improve[p] = (1 - p_p[p] / p_b[p]) * 100
        else:
            pct_improve[p] = 0

    # merged 供后续分析
    merged = pd.merge(
        df_b[["chat_id", "turn", "ttft_ms"]].copy(),
        df_p[["chat_id", "turn", "ttft_ms"]].copy(),
        on=["chat_id", "turn"],
        suffixes=("_baseline", "_prefetch"),
    ) if not df_b.empty and not df_p.empty else pd.DataFrame()

    # Mean TTFT / Mean TPOT
    mean_ttft_b = df_b["ttft_ms"].mean() if not df_b.empty else 0
    mean_ttft_p = df_p["ttft_ms"].mean() if not df_p.empty else 0
    tpot_b = df_b["tpot_ms"].dropna() if not df_b.empty and "tpot_ms" in df_b.columns else pd.Series(dtype=float)
    tpot_p = df_p["tpot_ms"].dropna() if not df_p.empty and "tpot_ms" in df_p.columns else pd.Series(dtype=float)
    mean_tpot_b = tpot_b.mean() if len(tpot_b) > 0 else 0
    mean_tpot_p = tpot_p.mean() if len(tpot_p) > 0 else 0
    p50_tpot_b = np.percentile(tpot_b, 50) if len(tpot_b) > 0 else 0
    p50_tpot_p = np.percentile(tpot_p, 50) if len(tpot_p) > 0 else 0

    # cached_tokens
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

    prefetch_cpu_hit_rate = 0
    mean_prefetch_cached = 0
    if "prefetch_cached_tokens" in df_p.columns:
        df_prefetch = df_p[df_p["prefetch_cached_tokens"].notna()].copy()
        if not df_prefetch.empty:
            prefetch_cpu_hit_rate = (df_prefetch["prefetch_cached_tokens"] > 0).mean() * 100
            mean_prefetch_cached = df_prefetch["prefetch_cached_tokens"].mean()

    prefetch_cpu_hit_rate_b = 0
    mean_prefetch_cached_b = 0
    if "prefetch_cached_tokens" in df_b.columns:
        df_prefetch_b = df_b[df_b["prefetch_cached_tokens"].notna()].copy()
        if not df_prefetch_b.empty:
            prefetch_cpu_hit_rate_b = (df_prefetch_b["prefetch_cached_tokens"] > 0).mean() * 100
            mean_prefetch_cached_b = df_prefetch_b["prefetch_cached_tokens"].mean()

    baseline_cached_rate = 0
    if "cached_tokens" in df_b.columns:
        df_b_cached = df_b[df_b["cached_tokens"].notna()]
        if not df_b_cached.empty:
            baseline_cached_rate = (df_b_cached["cached_tokens"] > 0).mean() * 100

    plain_cached_rate = 0
    plain_avg_ratio = 0
    if three_way and "cached_tokens" in df_plain.columns:
        df_pl = df_plain[df_plain["cached_tokens"].notna()]
        if not df_pl.empty:
            plain_cached_rate = (df_pl["cached_tokens"] > 0).mean() * 100
        df_pl_ratio = df_plain[
            (df_plain["cached_tokens"].notna()) & (df_plain["prompt_tokens"] > 0)
        ]
        if not df_pl_ratio.empty and (df_pl_ratio["cached_tokens"] > 0).any():
            plain_avg_ratio = (
                df_pl_ratio["cached_tokens"] / df_pl_ratio["prompt_tokens"]
            ).mean() * 100

    prefetch_avg_ratio = 0
    if not df_p_cached.empty and (df_p_cached["cached_tokens"] > 0).any():
        prefetch_avg_ratio = (df_p_cached["cached_tokens"] / df_p_cached["prompt_tokens"]).replace([np.inf, -np.inf], 0).mean() * 100
    baseline_avg_ratio = 0
    if "cached_tokens" in df_b.columns and "prompt_tokens" in df_b.columns:
        df_b_ratio = df_b[(df_b["cached_tokens"].notna()) & (df_b["prompt_tokens"] > 0)]
        if not df_b_ratio.empty and (df_b_ratio["cached_tokens"] > 0).any():
            baseline_avg_ratio = (df_b_ratio["cached_tokens"] / df_b_ratio["prompt_tokens"]).mean() * 100

    mean_b = df_b["ttft_ms"].mean() if not df_b.empty else 0
    mean_p = df_p["ttft_ms"].mean() if not df_p.empty else 0
    mean_improve = (1 - mean_p / mean_b) * 100 if mean_b > 0 else 0
    mean_ttft_plain = df_plain["ttft_ms"].mean() if not df_plain.empty else 0
    tpot_plain = (
        df_plain["tpot_ms"].dropna()
        if not df_plain.empty and "tpot_ms" in df_plain.columns
        else pd.Series(dtype=float)
    )
    mean_tpot_plain = tpot_plain.mean() if len(tpot_plain) > 0 else 0
    p50_tpot_plain = np.percentile(tpot_plain, 50) if len(tpot_plain) > 0 else 0
    mean_plain = mean_ttft_plain

    mean_diff = 0
    if not merged.empty:
        merged["ttft_diff_ms"] = merged["ttft_ms_baseline"] - merged["ttft_ms_prefetch"]
        mean_diff = merged["ttft_diff_ms"].mean()

    # 逐 Turn 分析
    turn_lines = []
    if len(merged) > 0 and "turn" in merged.columns:
        for turn_val in sorted(merged["turn"].unique()):
            sub = merged[merged["turn"] == turn_val]
            tb = sub["ttft_ms_baseline"].mean()
            tp = sub["ttft_ms_prefetch"].mean()
            turn_lines.append(f"| Turn {turn_val} | {tb:.1f} | {tp:.1f} |")
    turn_table = "\n".join(turn_lines) if turn_lines else "| (无配对样本) | - | - |"

    title = (
        "# 普通 vLLM / Prefetch / Prefetch+PCIe 调度 评估报告\n"
        if three_way
        else "# Prefetch KV Cache A/B 评估报告\n"
    )

    if three_way:

        def _sched_vs_pref_pct(pct: int) -> float:
            if p_b[pct] > 0:
                return (1 - p_p[pct] / p_b[pct]) * 100
            return 0.0

        summary_section = [
            "## 2. 汇总\n",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 普通 vLLM 样本数 | {len(df_plain)} |",
            f"| +Prefetch 样本数 | {len(df_b)} |",
            f"| +Prefetch+PCIe 样本数 | {len(df_p)} |",
            f"| 配对样本数（+Prefetch vs +Prefetch+PCIe） | {len(merged)} |",
            f"| 平均 TTFT 变化（PCIe 相对 +Prefetch） | {mean_improve:+.1f}% |",
            f"| +Prefetch+PCIe cached_tokens 命中率 | {hit_rate:.1f}% |",
            f"| +Prefetch+PCIe CPU→GPU 触发率 | {prefetch_cpu_hit_rate:.1f}% |",
            "",
            "## 3. TTFT 分位数对比\n",
            "| 分位 | 普通 vLLM (ms) | +Prefetch (ms) | +Prefetch+PCIe (ms) | PCIe 相对 +Prefetch |",
            "|------|----------------|----------------|---------------------|---------------------|",
        ]
        for pct in percentiles:
            imp = _sched_vs_pref_pct(pct)
            summary_section.append(
                f"| P{pct} | {p_plain[pct]:.2f} | {p_b[pct]:.2f} | {p_p[pct]:.2f} | {imp:+.1f}% |"
            )
        summary_section.extend(
            [
                "",
                "## 4. Mean TTFT / Mean TPOT 汇总\n",
                "| 指标 | 普通 vLLM | +Prefetch | +Prefetch+PCIe |",
                "|------|-----------|-----------|----------------|",
                f"| Mean TTFT (ms) | {mean_ttft_plain:.2f} | {mean_ttft_b:.2f} | {mean_ttft_p:.2f} |",
                f"| Mean TPOT (ms) | {mean_tpot_plain:.2f} | {mean_tpot_b:.2f} | {mean_tpot_p:.2f} |",
                f"| P50 TTFT (ms) | {p_plain[50]:.2f} | {p_b[50]:.2f} | {p_p[50]:.2f} |",
                f"| P50 TPOT (ms) | {p50_tpot_plain:.2f} | {p50_tpot_b:.2f} | {p50_tpot_p:.2f} |",
                "",
                "## 5. TTFT 详细统计\n",
                "| 模式 | Mean | P50 | P90 | P95 | P99 |",
                "|------|------|-----|-----|-----|-----|",
                f"| 普通 vLLM | {mean_plain:.2f} | {p_plain[50]:.2f} | {p_plain[90]:.2f} | {p_plain[95]:.2f} | {p_plain[99]:.2f} |",
                f"| +Prefetch | {mean_b:.2f} | {p_b[50]:.2f} | {p_b[90]:.2f} | {p_b[95]:.2f} | {p_b[99]:.2f} |",
                f"| +Prefetch+PCIe | {mean_p:.2f} | {p_p[50]:.2f} | {p_p[90]:.2f} | {p_p[95]:.2f} | {p_p[99]:.2f} |",
                "",
                "## 6. Prefix Cache\n",
                "| 指标 | 普通 vLLM | +Prefetch | +Prefetch+PCIe |",
                "|------|-----------|-----------|----------------|",
                f"| cached_tokens>0 占比 (%) | {plain_cached_rate:.1f} | {baseline_cached_rate:.1f} | {hit_rate:.1f} |",
                f"| 平均 cached/prompt (%) | {plain_avg_ratio:.1f} | {baseline_avg_ratio:.1f} | {prefetch_avg_ratio:.1f} |",
                f"| prefetch_cached_tokens>0 (%) | — | {prefetch_cpu_hit_rate_b:.1f} | {prefetch_cpu_hit_rate:.1f} |",
                f"| 平均 prefetch_cached_tokens | — | {mean_prefetch_cached_b:.1f} | {mean_prefetch_cached:.1f} |",
                "",
                "## 7. 配对对比（+Prefetch vs +Prefetch+PCIe）\n",
                f"- 配对样本数: {len(merged)}",
                f"- 平均 TTFT 差值 (+Prefetch − +Prefetch+PCIe): {mean_diff:.2f} ms",
                "",
                "## 8. 逐 Turn 分析（+Prefetch vs +Prefetch+PCIe）\n",
                "| Turn | +Prefetch TTFT (ms) | +Prefetch+PCIe TTFT (ms) |",
                "|------|----------------------|---------------------------|",
                turn_table,
                "",
            ]
        )
        md_parts = [
            title,
            "## 1. 配置详情\n",
            _format_config_section("实验配置", config_str),
            _format_config_section("vLLM 配置", args.vllm_config),
            _format_config_section("测试配置", args.test_config),
            *summary_section,
        ]
    else:
        md_parts = [
            title,
            "## 1. 配置详情\n",
            _format_config_section("实验配置", config_str),
            _format_config_section("vLLM 配置", args.vllm_config),
            _format_config_section("测试配置", args.test_config),
            "## 2. 汇总\n",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| Baseline 样本数 | {len(df_b)} |",
            f"| Prefetch 样本数 | {len(df_p)} |",
            f"| 配对样本数 | {len(merged)} |",
            f"| 平均 TTFT 降低 | {mean_improve:.1f}% |",
            f"| Prefetch cached_tokens 命中率 | {hit_rate:.1f}% |",
            f"| Prefetch CPU→GPU 触发率 | {prefetch_cpu_hit_rate:.1f}% |",
            "",
            "## 3. TTFT 分位数对比\n",
            "| 分位 | Baseline (ms) | Prefetch (ms) | TTFT 提升 |",
            "|------|---------------|---------------|-----------|",
            f"| P50 | {p_b[50]:.2f} | {p_p[50]:.2f} | {pct_improve[50]:.1f}% |",
            f"| P90 | {p_b[90]:.2f} | {p_p[90]:.2f} | {pct_improve[90]:.1f}% |",
            f"| P95 | {p_b[95]:.2f} | {p_p[95]:.2f} | {pct_improve[95]:.1f}% |",
            f"| P99 | {p_b[99]:.2f} | {p_p[99]:.2f} | {pct_improve[99]:.1f}% |",
            "",
            "## 4. Mean TTFT / Mean TPOT 汇总\n",
            "| 指标 | Baseline | Prefetch | 提升 |",
            "|------|----------|----------|------|",
            f"| Mean TTFT (ms) | {mean_ttft_b:.2f} | {mean_ttft_p:.2f} | {(1 - mean_ttft_p / mean_ttft_b) * 100:.1f}% |" if mean_ttft_b > 0 else "| Mean TTFT (ms) | - | - | - |",
            f"| Mean TPOT (ms) | {mean_tpot_b:.2f} | {mean_tpot_p:.2f} | {(1 - mean_tpot_p / mean_tpot_b) * 100:.1f}% |" if mean_tpot_b > 0 else "| Mean TPOT (ms) | - | - | - |",
            f"| P50 TTFT (ms) | {p_b[50]:.2f} | {p_p[50]:.2f} | {pct_improve[50]:.1f}% |",
            f"| P50 TPOT (ms) | {p50_tpot_b:.2f} | {p50_tpot_p:.2f} | - |",
            "",
            "## 5. TTFT 详细统计\n",
            "| 模式 | Mean | P50 | P90 | P95 | P99 |",
            "|------|------|-----|-----|-----|-----|",
            f"| Baseline | {mean_b:.2f} | {p_b[50]:.2f} | {p_b[90]:.2f} | {p_b[95]:.2f} | {p_b[99]:.2f} |",
            f"| Prefetch | {mean_p:.2f} | {p_p[50]:.2f} | {p_p[90]:.2f} | {p_p[95]:.2f} | {p_p[99]:.2f} |",
            "",
            "## 6. Prefix Cache\n",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| Prefetch cached_tokens 命中率 | {hit_rate:.1f}% |",
            f"| Prefetch 平均 cached_tokens/prompt_tokens | {prefetch_avg_ratio:.1f}% |",
            f"| Baseline 平均 cached_tokens/prompt_tokens | {baseline_avg_ratio:.1f}% |",
            f"| Prefetch CPU→GPU load 触发率 | {prefetch_cpu_hit_rate:.1f}% |",
            f"| 平均 prefetch_cached_tokens | {mean_prefetch_cached:.1f} |",
            "",
            "## 7. 配对对比\n",
            f"- 配对样本数: {len(merged)}",
            f"- 平均 TTFT 差值 (baseline - prefetch): {mean_diff:.2f} ms",
            "",
            "## 8. 逐 Turn 分析\n",
            "| Turn | Baseline TTFT (ms) | Prefetch TTFT (ms) |",
            "|------|---------------------|---------------------|",
            turn_table,
            "",
        ]

    md = "\n".join(md_parts)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"报告已生成: {args.output}")


if __name__ == "__main__":
    main()
