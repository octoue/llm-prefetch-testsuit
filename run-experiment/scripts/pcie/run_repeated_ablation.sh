#!/bin/bash
# 自动化重复消融实验脚本
#
# 调用 auto_run_pcie_ablation_ab.sh N 次，每次产生独立的实验结果，
# 最后生成跨轮次汇总报告（mean ± std）。
#
# 用法:
#   ./run_repeated_ablation.sh [dataset] [options]
#
# 示例:
#   ./run_repeated_ablation.sh pcie-heavy --qps 3.0 --repeats 3
#   ./run_repeated_ablation.sh pcie-heavy --qps 3.0 --repeats 5 --gpu-blocks 1000
#
# 产物:
#   results/ 下会产生 N 个独立的 ablation 实验目录（由 auto_run_pcie_ablation_ab.sh 生成）
#   results/<dataset>_repeated_q<QPS>_r<N>/
#     summary.md              (跨轮次汇总)
#     run_dirs.txt            (各轮次实验目录列表)
#
# 挂机运行:
#   nohup bash scripts/pcie/run_repeated_ablation.sh pcie-heavy --qps 3.0 --repeats 3 > repeated.log 2>&1 &

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$RUN_EXP_DIR"

source config/system.env
source config/datasets.env
source config/experiments.env

# ========== 参数解析 ==========
DATASET="${1:-pcie-heavy}"
shift 2>/dev/null || true

REPEATS=3
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeats)  REPEATS="$2"; shift 2 ;;
        # 其他参数原样传给 auto_run_pcie_ablation_ab.sh
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# 从 EXTRA_ARGS 中提取 QPS 和 lead-time 用于目录命名
QPS_FOR_NAME="$QPS"  # 默认从 experiments.env
LEAD_FOR_NAME="$PREFETCH_LEAD_TIME"
for i in "${!EXTRA_ARGS[@]}"; do
    [[ "${EXTRA_ARGS[$i]}" == "--qps" ]] && QPS_FOR_NAME="${EXTRA_ARGS[$((i+1))]}"
    [[ "${EXTRA_ARGS[$i]}" == "--lead-time" ]] && LEAD_FOR_NAME="${EXTRA_ARGS[$((i+1))]}"
done

REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUMMARY_DIR="$REPO_ROOT/results/${DATASET}_repeated_q${QPS_FOR_NAME}_r${REPEATS}"
mkdir -p "$SUMMARY_DIR"

echo "============================================"
echo "Repeated Ablation Experiment"
echo "============================================"
echo "Dataset:    $DATASET"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "Repeats:    $REPEATS"
echo "Summary:    $SUMMARY_DIR"
echo "Start time: $(date)"
echo "============================================"
echo ""

TOTAL_START=$(date +%s)
RUN_DIRS=()
FAILED=()

for RUN_NUM in $(seq 1 "$REPEATS"); do
    echo ""
    echo "========================================================"
    echo "  Repeat $RUN_NUM / $REPEATS  ($(date))"
    echo "========================================================"

    if bash "$SCRIPT_DIR/auto_run_pcie_ablation_ab.sh" "$DATASET" "${EXTRA_ARGS[@]}"; then
        # auto_run 生成的目录是最新的 ablation 目录
        LATEST_DIR=$(ls -dt "$REPO_ROOT/results/"*ablation_${DATASET}_q${QPS_FOR_NAME}* 2>/dev/null | head -1)
        if [[ -n "$LATEST_DIR" && -d "$LATEST_DIR" ]]; then
            RUN_DIRS+=("$LATEST_DIR")
            echo "✓ Repeat $RUN_NUM completed: $LATEST_DIR"
        else
            echo "⚠️  Repeat $RUN_NUM: cannot find results directory"
            FAILED+=("$RUN_NUM")
        fi
    else
        echo "❌ Repeat $RUN_NUM FAILED"
        FAILED+=("$RUN_NUM")
    fi
done

TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - TOTAL_START) / 60 ))

# 保存目录列表
printf '%s\n' "${RUN_DIRS[@]}" > "$SUMMARY_DIR/run_dirs.txt"

# ========== 汇总报告 ==========
echo ""
echo "============================================"
echo "Generating summary across ${#RUN_DIRS[@]} runs..."
echo "============================================"

python3 -c "
import json, numpy as np, os, sys

run_dirs = '''$(printf '%s\n' "${RUN_DIRS[@]}")'''.strip().split('\n')
run_dirs = [d for d in run_dirs if d]
n_runs = len(run_dirs)

if n_runs == 0:
    print('No successful runs to summarize.')
    sys.exit(0)

groups = ['g3_full_sched', 'g2_sched_no_phase', 'g1_prefetch_only', 'g0_no_prefetch']
labels = {'g3_full_sched': 'G3 (+Phase-Aware)', 'g2_sched_no_phase': 'G2 (+Scheduler)',
          'g1_prefetch_only': 'G1 (+Prefetch)', 'g0_no_prefetch': 'G0 (Baseline)'}

results = {g: {'ttft_means': [], 'ttft_p50s': [], 'ttft_p95s': [], 'ttft_p99s': [],
                'tpot_means': [], 'tpot_p99s': []} for g in groups}

for run_dir in run_dirs:
    for g in groups:
        path = os.path.join(run_dir, f'prefetch_{g}.jsonl')
        if not os.path.exists(path):
            print(f'⚠️  Missing: {path}', file=sys.stderr)
            continue
        ttfts, tpots = [], []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d.get('success') and d.get('ttft_ms'):
                    ttfts.append(d['ttft_ms'])
                if d.get('success') and d.get('tpot_ms') and d['tpot_ms'] > 0:
                    tpots.append(d['tpot_ms'])
        if ttfts:
            t = np.array(ttfts)
            results[g]['ttft_means'].append(t.mean())
            results[g]['ttft_p50s'].append(np.median(t))
            results[g]['ttft_p95s'].append(np.percentile(t, 95))
            results[g]['ttft_p99s'].append(np.percentile(t, 99))
        if tpots:
            p = np.array(tpots)
            results[g]['tpot_means'].append(p.mean())
            results[g]['tpot_p99s'].append(np.percentile(p, 99))

# Generate markdown
lines = ['# Repeated Ablation Summary', '',
         f'**Dataset**: $DATASET | **QPS**: $QPS_FOR_NAME | **Repeats**: {n_runs}', '',
         f'Run directories:']
for i, d in enumerate(run_dirs):
    lines.append(f'- Run {i+1}: \`{os.path.basename(d)}\`')

lines += ['', '## Mean TTFT (ms) per run', '',
          '| Group | ' + ' | '.join(f'Run {i+1}' for i in range(n_runs)) + ' | **Mean ± Std** |',
          '|-------|' + '|'.join(['------'] * n_runs) + '|------------|']

for g in groups:
    vals = results[g]['ttft_means']
    if vals:
        row = f'| {labels[g]} | ' + ' | '.join(f'{v:.0f}' for v in vals)
        row += f' | **{np.mean(vals):.0f} ± {np.std(vals):.0f}** |'
        lines.append(row)

lines += ['', '## TTFT Percentiles (mean ± std across runs)', '',
          '| Group | Mean TTFT | P50 | P95 | P99 | TPOT Mean | TPOT P99 |',
          '|-------|-----------|-----|-----|-----|-----------|----------|']

def fmt(arr):
    if not arr: return '-'
    return f'{np.mean(arr):.0f}±{np.std(arr):.0f}'

def fmt1(arr):
    if not arr: return '-'
    return f'{np.mean(arr):.1f}±{np.std(arr):.1f}'

for g in groups:
    r = results[g]
    if r['ttft_means']:
        lines.append(f'| {labels[g]} | {fmt(r[\"ttft_means\"])} | {fmt(r[\"ttft_p50s\"])} | '
                     f'{fmt(r[\"ttft_p95s\"])} | {fmt(r[\"ttft_p99s\"])} | '
                     f'{fmt1(r[\"tpot_means\"])} | {fmt1(r[\"tpot_p99s\"])} |')

# Improvements
lines += ['', '## Improvements vs G0 (mean across runs)', '']
if results['g0_no_prefetch']['ttft_means']:
    g0m = np.mean(results['g0_no_prefetch']['ttft_means'])
    for g in ['g1_prefetch_only', 'g2_sched_no_phase', 'g3_full_sched']:
        if results[g]['ttft_means']:
            gm = np.mean(results[g]['ttft_means'])
            pct = (g0m - gm) / g0m * 100
            lines.append(f'- {labels[g]} vs G0: **{pct:.1f}%**')

    lines += ['', '## Incremental improvements (mean across runs)', '']
    g1m = np.mean(results['g1_prefetch_only']['ttft_means']) if results['g1_prefetch_only']['ttft_means'] else 0
    g2m = np.mean(results['g2_sched_no_phase']['ttft_means']) if results['g2_sched_no_phase']['ttft_means'] else 0
    g3m = np.mean(results['g3_full_sched']['ttft_means']) if results['g3_full_sched']['ttft_means'] else 0
    if g1m > 0:
        lines.append(f'- G3 vs G1 (full scheduling contribution): **{(g1m-g3m)/g1m*100:.1f}%**')
    if g2m > 0:
        lines.append(f'- G3 vs G2 (Phase-Aware contribution): **{(g2m-g3m)/g2m*100:.1f}%**')

summary_path = '$SUMMARY_DIR/summary.md'
with open(summary_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('\n'.join(lines))
" 2>&1

echo ""
echo "============================================"
echo "✅ All $REPEATS repeats completed in ${TOTAL_MIN}m"
echo "   Successful: ${#RUN_DIRS[@]} / $REPEATS"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "   Failed: ${FAILED[*]}"
fi
echo "   Summary: $SUMMARY_DIR/summary.md"
echo "   Run list: $SUMMARY_DIR/run_dirs.txt"
echo "============================================"
