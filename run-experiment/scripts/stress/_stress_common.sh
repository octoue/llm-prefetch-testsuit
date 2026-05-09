#!/bin/bash
# Helpers shared by the three stress runners.
# Source this from run_stress_s{1,2,3}.sh.

set -e

# Path layout (this file lives at run-experiment/scripts/stress/_stress_common.sh):
#   SCRIPT_DIR     = .../run-experiment/scripts/stress
#   RUN_EXP_DIR    = .../run-experiment
#   TESTSUIT_ROOT  = .../llm-prefetch-testsuit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TESTSUIT_ROOT="$(cd "$RUN_EXP_DIR/.." && pwd)"
UTILS_DIR="$RUN_EXP_DIR/scripts/utils"

# Sanity-check critical files up-front so we fail loudly instead of
# silently turning every step into a no-op.
for f in \
    "$UTILS_DIR/sample_power.sh" \
    "$UTILS_DIR/sample_perf.sh" \
    "$UTILS_DIR/sample_pcie.sh" \
    "$RUN_EXP_DIR/prefetch_stress_runner.py" \
    "$RUN_EXP_DIR/config/system.env"
do
  [ -e "$f" ] || { echo "FATAL: required file missing: $f" >&2; exit 1; }
done

# Load global config (system.env now guaranteed to exist by the check above).
set -a
source "$RUN_EXP_DIR/config/system.env"
[ -f "$RUN_EXP_DIR/config/datasets.env" ] && source "$RUN_EXP_DIR/config/datasets.env"
set +a

# DATA_ROOT in datasets.env is "../data" (relative to run-experiment/).
# Resolve to an absolute path so subsequent commands are CWD-independent.
if [ -n "$DATA_ROOT" ] && [ "${DATA_ROOT#/}" = "$DATA_ROOT" ]; then
  DATA_ROOT="$(cd "$RUN_EXP_DIR/$DATA_ROOT" && pwd)"
fi
DATA_ROOT="${DATA_ROOT:-$TESTSUIT_ROOT/data}"

API_PORT="${API_PORT:-8000}"
API_BASE="${API_BASE:-http://localhost:${API_PORT}/v1}"
MODEL="${STRESS_MODEL:-${MODEL_PATH}}"
[[ -z "$MODEL" ]] && {
  echo "FATAL: MODEL not set (export STRESS_MODEL or set MODEL_PATH in system.env)" >&2
  exit 1
}

# S2/S3 background and S1 attack share the same real metadata trace by
# default; the runner derives attack prefixes from its multi-turn structure.
# Override ATTACK_PREFIX_FILE to point at a content JSONL for fully synthetic
# baselines (rare; mostly for ablation against the metadata-derived path).
DEFAULT_BG_TRACE="$DATA_ROOT/pcie_stress_heavy.jsonl"
DEFAULT_ATTACK_PREFIX="$DEFAULT_BG_TRACE"
[ -e "$DEFAULT_BG_TRACE" ] || {
  echo "FATAL: heavy trace missing: $DEFAULT_BG_TRACE" >&2
  echo "  generate it via run-experiment/scripts/prefetch/run_experiment.sh pcie-heavy" >&2
  exit 1
}

# Run the requested stress scenario together with power / pcie sampling,
# producing a self-contained results directory.
#
# Args (positional, all required):
#   $1 results_dir
#   $2 scenario  s1|s2|s3
# Optional env vars consumed:
#   DURATION_SEC, REPEAT, RATIO_OVERRIDE, TTL_OVERRIDE
#   PREFETCH_QPS, QPS, BURST_SIZE, BURST_INTERVAL_MS, MIX_RATIO
run_one_unit() {
  local results_dir="$1"; shift
  local scenario="$1"; shift
  local duration="${DURATION_SEC:-180}"

  mkdir -p "$results_dir"

  echo "==> stress unit: scenario=$scenario duration=${duration}s"
  echo "    results -> $results_dir"

  # Reset prefix cache so each run starts from a clean slate.
  curl -s -X POST "http://localhost:${API_PORT}/reset_prefix_cache?reset_external=true" \
       > /dev/null || true
  sleep 3

  # Start PCIe tracer (clears in-memory event buffer); on /stop_profile the
  # engine flushes events to <profiler_dir>/pcie_events_<rank>.json.
  curl -s -X POST "http://localhost:${API_PORT}/start_profile" >/dev/null || true

  # Read the GPU IDs the stress server acquired (written by
  # start_vllm_stress.sh) so power sampling stays scoped to our cards.
  local stress_gpu_ids=""
  if [ -f "$RUN_EXP_DIR/.stress_gpus" ]; then
    stress_gpu_ids=$(cat "$RUN_EXP_DIR/.stress_gpus" 2>/dev/null || true)
  fi

  # Background samplers.
  bash "$UTILS_DIR/sample_power.sh" \
        "$results_dir/power.csv" "$duration" "$stress_gpu_ids" &
  local power_pid=$!
  bash "$UTILS_DIR/sample_perf.sh" \
        "$results_dir/perf_pkg.txt" "$duration" &
  local perf_pid=$!

  # Common runner CLI assembly.
  local extra_args=()
  case "$scenario" in
    s1)
      extra_args+=( --prefetch-qps "${PREFETCH_QPS:-200}" \
                    --attack-prefix-file "${ATTACK_PREFIX_FILE:-$DEFAULT_ATTACK_PREFIX}" )
      ;;
    s2)
      extra_args+=( --bg-trace-file "${BG_TRACE_FILE:-$DEFAULT_BG_TRACE}" \
                    --qps "${QPS:-1.0}" \
                    --burst-size "${BURST_SIZE:-5}" \
                    --burst-interval-ms "${BURST_INTERVAL_MS:-50}" \
                    --abandon-prob "${ABANDON_PROB:-1.0}" )
      ;;
    s3)
      extra_args+=( --bg-trace-file "${BG_TRACE_FILE:-$DEFAULT_BG_TRACE}" \
                    --attack-prefix-file "${ATTACK_PREFIX_FILE:-$DEFAULT_ATTACK_PREFIX}" \
                    --qps "${QPS:-1.0}" \
                    --mix-ratio "${MIX_RATIO:-1.0}" )
      ;;
  esac

  # === Baseline workload ============================================
  # Launch the user's existing prefetch_ab_runner as a background load.
  # This is the baseline experiment from chapter 3; it sends real
  # multi-turn requests with prefetches, naturally populating the GPU
  # prefix cache and CPU offload cache. The stress overlay below then
  # lands on top of a realistic cache state, so attack/burst prefetches
  # actually exercise the CPU_HIT path (and thereby admission control,
  # quota and TTL reclaim) rather than discarding everything via NO_HIT.
  echo "    launching baseline (prefetch_ab_runner) at qps=${BASELINE_QPS:-1.0}, conv=${BASELINE_NUM_CONV:-40}"
  python3 "$RUN_EXP_DIR/prefetch_ab_runner.py" \
      --trace-file "$DEFAULT_BG_TRACE" \
      --mode prefetch \
      --qps "${BASELINE_QPS:-1.0}" \
      --num-multi-turn "${BASELINE_NUM_CONV:-40}" \
      --model "$MODEL" \
      --api-base "$API_BASE" \
      --output "$results_dir/baseline.jsonl" \
      --timeout "$duration" \
      --schedule-mode "${SCHEDULE_MODE:-scaled-timestamp}" \
      &> "$results_dir/baseline.log" &
  local baseline_pid=$!

  # Let baseline get a head start so the cache has content before the
  # attack hits. Default 15s is enough to schedule a few turns through
  # real inference and start spilling KV to CPU offload.
  local head_start="${BASELINE_HEAD_START:-15}"
  echo "    baseline head start: ${head_start}s"
  sleep "$head_start"

  # === Stress overlay ===============================================
  # Run the attack/burst runner for the remaining duration.
  local overlay_dur=$(( duration - head_start ))
  if [ "$overlay_dur" -lt 30 ]; then overlay_dur=30; fi
  echo "    launching $scenario overlay for ${overlay_dur}s"
  local rc=0
  python3 "$RUN_EXP_DIR/prefetch_stress_runner.py" \
      --scenario "$scenario" \
      --api-base "$API_BASE" \
      --model "$MODEL" \
      --duration-sec "$overlay_dur" \
      --output-jsonl "$results_dir/requests.jsonl" \
      --output-summary "$results_dir/summary.json" \
      "${extra_args[@]}" \
      &> "$results_dir/runner.log" || rc=$?
  echo "$rc" > "$results_dir/runner.exit_code"
  if [ "$rc" -ne 0 ]; then
    echo "    WARN: overlay runner exited with code $rc; see $results_dir/runner.log" >&2
  fi

  # Wait for baseline to finish (it has its own --timeout matching duration).
  wait "$baseline_pid" 2>/dev/null || true

  # Stop samplers.
  kill -TERM "$power_pid" 2>/dev/null || true
  kill -TERM "$perf_pid" 2>/dev/null || true
  wait "$power_pid" 2>/dev/null || true
  wait "$perf_pid" 2>/dev/null || true

  # Flush PCIe tracer to disk; sample_pcie.sh then merges & moves files aside.
  curl -s -X POST "http://localhost:${API_PORT}/stop_profile" >/dev/null || true
  sleep 2
  bash "$UTILS_DIR/sample_pcie.sh" "$results_dir/pcie_events.json" \
       "${PCIE_PROFILER_DIR:-$RUN_EXP_DIR/profiler_output}" || true

  # Quick post-hoc sanity check of artefacts.
  for required in summary.json requests.jsonl baseline.jsonl power.csv pcie_events.json; do
    if [ ! -s "$results_dir/$required" ]; then
      echo "    WARN: $results_dir/$required is empty or missing" >&2
    fi
  done
  echo "    artefacts: summary.json (overlay) | baseline.jsonl (real load) | pcie_events.json | power.csv"
  sleep 30
}
