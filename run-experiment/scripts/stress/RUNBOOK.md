# Prefetch Stress Experiments — Runbook

Targets the experiment matrix E1–E6 in
`/Users/duck/Documents/code/claude-docs/review/20260507_stress_experiment_plan.md`.

Hardware: A100-80GB × 2 (32B) or × 4 (72B). vLLM is built from the local
`/Users/duck/Documents/code/vllm` checkout (TTL + Prometheus prefetch
counters already wired in this branch).

## 0. Once per machine

```bash
# Build / install the patched vLLM (or activate an existing venv).
cd /path/to/vllm && pip install -e .

# Generate the long-prefix templates used by S1/S3.
cd /path/to/llm-prefetch-testsuit
python3 data/generate_attack_prefixes.py \
    --output data/synth_attack_prefix_8k.jsonl \
    --num 30 --target-tokens 7800

# (Optional) tune kernel for low-latency / disable extraneous GPU users.
sudo tuned-adm profile latency-performance || true
```

## 1. Start the stress server

Default: Qwen2.5-32B on TP=2 (A100 × 2). For 72B set `--tp 4`.

```bash
cd /path/to/llm-prefetch-testsuit/run-experiment
bash scripts/stress/start_vllm_stress.sh \
    --tp 2 \
    --ratio 0.3 \
    --ttl-ms 60000 \
    --port 8000
```

Key flags exposed by the patched engine:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--max-prefetch-block-ratio` | 0.3 | Prefetch quota ratio (E2 sweep) |
| `--prefetch-ttl-ms` | 60000 | TTL for unconsumed prefetch blocks (E3 ablation) |
| `--prefetch-block-threshold` | 150 | Free-block threshold below which prefetches defer |

For ratio / TTL ablations the simplest path is to **restart** the server
between values. The runbook scripts honour env vars `STRESS_BLOCK_RATIO`
and `STRESS_PREFETCH_TTL_MS` when calling `start_vllm_stress.sh`.

## 2. E1 — S1 strength sweep (mandatory)

Three repeats × three QPS levels at the default ratio.

```bash
PREFETCH_QPS_LIST="50 200 800" \
RATIO_LIST="0.3" \
DURATION_SEC=300 \
REPEAT=3 \
bash scripts/stress/run_stress_s1.sh
```

## 3. E2 — quota ablation (mandatory)

Restart the server between ratios:

```bash
for r in 0 0.1 0.3 0.5; do
  pkill -f 'vllm serve' || true; sleep 5
  STRESS_BLOCK_RATIO=$r \
    bash scripts/stress/start_vllm_stress.sh --ratio "$r" &
  # Wait until /health returns 200, then run:
  PREFETCH_QPS_LIST="200" RATIO_LIST="$r" REPEAT=3 \
    bash scripts/stress/run_stress_s1.sh "results/stress/e2_ratio${r}"
done
```

## 4. E3 — S2 misclick burst + TTL ablation (mandatory)

```bash
# Default: TTL=60s, sweep N.
BURST_SIZE_LIST="2 5 10" TTL_LIST="60000" REPEAT=3 \
  bash scripts/stress/run_stress_s2.sh

# TTL ablation: re-run with TTL=0 (TTL disabled) to compare.
pkill -f 'vllm serve' || true; sleep 5
STRESS_PREFETCH_TTL_MS=0 \
  bash scripts/stress/start_vllm_stress.sh --ttl-ms 0 &
BURST_SIZE_LIST="2 5 10" TTL_LIST="0" REPEAT=3 \
  bash scripts/stress/run_stress_s2.sh "results/stress/e3_ttl0"
```

## 5. E4 — S3 mixed background (mandatory)

```bash
MIX_RATIO_LIST="0.25 1.0 4.0" REPEAT=3 \
  bash scripts/stress/run_stress_s3.sh
```

## 6. E5 — 72B replication (optional)

```bash
pkill -f 'vllm serve' || true; sleep 5
STRESS_MODEL=/path/to/Qwen2.5-72B \
  bash scripts/stress/start_vllm_stress.sh --tp 4 --ratio 0.3 --ttl-ms 60000 &
MIX_RATIO_LIST="1.0" REPEAT=2 DURATION_SEC=600 \
  bash scripts/stress/run_stress_s3.sh "results/stress/e5_72b"
```

## 7. E6 — long-duration steady state (optional)

```bash
PREFETCH_QPS_LIST="200" RATIO_LIST="0.3" \
DURATION_SEC=1800 REPEAT=1 \
  bash scripts/stress/run_stress_s1.sh "results/stress/e6_steady"
```

## 8. Aggregate & plot

```bash
for d in results/stress/s1_* results/stress/s2_* results/stress/s3_* \
         results/stress/e2_* results/stress/e3_* results/stress/e5_* \
         results/stress/e6_*; do
  [ -d "$d" ] || continue
  python3 ../result-analysis/analyze_stress.py --root "$d"
done
```

Each `--root` produces `aggregate.csv` and `figs/{power_curve,wasted_io,ttft_cdf}.pdf`.

## 9. What lands in the paper

- **`figures/prefetch/stress_s1_power.pdf`** ← `figs/power_curve.pdf` of the E1 root.
- **`figures/prefetch/stress_s1_wasted_io.pdf`** ← `figs/wasted_io.pdf` of E1+E2 combined.
- **`figures/prefetch/stress_s3_ttft_cdf.pdf`** ← `figs/ttft_cdf.pdf` of E4.

Numerical claims for the new `\subsection{极端预取失败场景下的开销分析}`
draft come from `aggregate.csv`:

- "废弃 I/O 比例" → `prom_no_hits + prom_expired` ÷ all `prom_*` from S1.
- "活跃推理 TTFT 退化幅度" → `real_ttft_p95_ms` from S3 mix=1.0 vs mix=0.25.
- "GPU 平均功耗增量" → `gpu_mean_w` of S1 vs `gpu_mean_w` of an idle baseline run.

## 10. Troubleshooting

| Symptom | Probable cause | Fix |
| --- | --- | --- |
| `prom_*` all zero in summary.json | server lacks the new Prometheus counters | rebuild vLLM from this branch |
| `pcie_events.json` empty | `VLLM_PCIE_TRACE` not honoured / `start_profile` failed | confirm `start_vllm_stress.sh` exported the env, retry |
| S1 latency near 0 ms but `cached_tokens=N` | prefetch hit GPU prefix cache instead of allocating | increase number of attack prefixes or shorten attack-prefix TTL |
| S2 real-request TTFT looks identical to S1 | benign trace exhausted before window ended | raise `--max-chains-records` or repeat with fresh trace |
