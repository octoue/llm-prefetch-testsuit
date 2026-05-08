#!/bin/bash
# Collect PCIeTracer events that the server flushed on /stop_profile.
#
# In start_vllm_stress.sh we pass --profiler-config '{"profiler":"pcie",
# "torch_profiler_dir":"<dir>"}'. On /stop_profile the engine writes
# <dir>/pcie_events_<rank>.json (one per worker).
#
# This script merges those JSON arrays into a single file by collating the
# events list across ranks. Existing files are then renamed (so the next
# /stop_profile starts clean).
#
# Usage:
#   sample_pcie.sh <output_json> [profiler_dir]

set -e

OUT="${1:?usage: sample_pcie.sh <output_json> [profiler_dir]}"
DIR="${2:-${PCIE_PROFILER_DIR:-$(pwd)/profiler_output}}"

mkdir -p "$(dirname "$OUT")"

if [[ ! -d "$DIR" ]]; then
  echo "[]" > "$OUT"
  echo "WARN: profiler dir $DIR not found; wrote empty array"
  exit 0
fi

shopt -s nullglob
files=( "$DIR"/pcie_events_*.json )
if [[ ${#files[@]} -eq 0 ]]; then
  echo "[]" > "$OUT"
  echo "WARN: no pcie_events_*.json under $DIR; wrote empty array"
  exit 0
fi

python3 - "$OUT" "${files[@]}" <<'PY'
import json, sys
out = sys.argv[1]
events = []
for path in sys.argv[2:]:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        continue
    if isinstance(data, list):
        events.extend(data)
with open(out, "w") as f:
    json.dump(events, f)
print(f"merged {len(sys.argv) - 2} files -> {out} ({len(events)} events)")
PY

# Move consumed files aside so the next stop_profile starts clean.
ts=$(date +%Y%m%d_%H%M%S)
for f in "${files[@]}"; do
  mv "$f" "${f%.json}.${ts}.consumed.json"
done
