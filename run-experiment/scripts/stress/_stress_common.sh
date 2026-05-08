#!/bin/bash
# Helpers shared by the three stress runners.
# Source this from run_stress_s{1,2,3}.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTSUIT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_EXP_DIR="$TESTSUIT_ROOT/run-experiment"
UTILS_DIR="$RUN_EXP_DIR/scripts/utils"

# Load global config so MODEL_PATH / API_PORT etc are available.
[ -f "$RUN_EXP_DIR/config/system.env" ] && source "$RUN_EXP_DIR/config/system.env"
[ -f "$RUN_EXP_DIR/config/datasets.env" ] && source "$RUN_EXP_DIR/config/datasets.env"

API_PORT="${API_PORT:-8000}"
API_BASE="${API_BASE:-http://localhost:${API_PORT}/v1}"
MODEL="${STRESS_MODEL:-${MODEL_PATH}}"
# S2/S3 background and S1 attack share the same real metadata trace by
# default; the runner derives attack prefixes from its multi-turn structure.
# Override ATTACK_PREFIX_FILE to point at a content JSONL for fully synthetic
# baselines (rare; mostly for ablation against the metadata-derived path).
DEFAULT_BG_TRACE="${DATA_ROOT:-$TESTSUIT_ROOT/data}/pcie_stress_heavy.jsonl"
DEFAULT_ATTACK_PREFIX="$DEFAULT_BG_TRACE"

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

  # Background samplers.
  bash "$UTILS_DIR/sample_power.sh" \
        "$results_dir/power.csv" "$duration" &
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

  python3 "$RUN_EXP_DIR/prefetch_stress_runner.py" \
      --scenario "$scenario" \
      --api-base "$API_BASE" \
      --model "$MODEL" \
      --duration-sec "$duration" \
      --output-jsonl "$results_dir/requests.jsonl" \
      --output-summary "$results_dir/summary.json" \
      "${extra_args[@]}" \
      &> "$results_dir/runner.log" || true

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

  echo "    summary written to $results_dir/summary.json"
  sleep 30
}
