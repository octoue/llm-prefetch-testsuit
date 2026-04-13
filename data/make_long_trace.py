#!/usr/bin/env python3
"""
Expand a jsonl trace by repeating it N times with offset chat_ids so that
each pass appears as disjoint conversations.

Simply `cat`-ing the trace N times would break the experiment: chat_ids
would collide, parent_chat_id links would become ambiguous, and
prefetch_ab_runner's multi-turn tree builder would get confused.

This script:
  - offsets chat_id and parent_chat_id by (pass_idx * stride) per pass,
    where stride = max_original_chat_id + 1
  - offsets timestamp so the passes don't overlap in wall-clock coordinates
    (kept for clarity; the runner paces by --qps anyway)

Usage:
    python3 make_long_trace.py <input.jsonl> <multiplier>

Example:
    python3 make_long_trace.py pcie_stress_heavy.jsonl 5
    # => writes pcie_stress_heavy_x5.jsonl
"""
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    try:
        multiplier = int(sys.argv[2])
    except ValueError:
        print(f"ERROR: multiplier must be an integer, got {sys.argv[2]!r}",
              file=sys.stderr)
        sys.exit(1)

    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if multiplier < 2:
        print("ERROR: multiplier must be >= 2", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("ERROR: input file is empty", file=sys.stderr)
        sys.exit(1)

    max_chat_id = max(r.get("chat_id", 0) for r in records)
    chat_id_stride = max_chat_id + 1

    max_timestamp = max(float(r.get("timestamp", 0.0)) for r in records)
    timestamp_stride = max_timestamp + 1.0  # 1s gap between passes

    original_conv_count = sum(
        1 for r in records if r.get("parent_chat_id", -1) == -1
    )

    output_path = input_path.with_name(
        input_path.stem + f"_x{multiplier}" + input_path.suffix
    )

    with open(output_path, "w") as out:
        for pass_idx in range(multiplier):
            chat_offset = pass_idx * chat_id_stride
            ts_offset = pass_idx * timestamp_stride
            for r in records:
                new_r = dict(r)
                new_r["chat_id"] = r["chat_id"] + chat_offset
                if r.get("parent_chat_id", -1) != -1:
                    new_r["parent_chat_id"] = r["parent_chat_id"] + chat_offset
                new_r["timestamp"] = float(r.get("timestamp", 0.0)) + ts_offset
                out.write(json.dumps(new_r) + "\n")

    new_conv_count = original_conv_count * multiplier
    new_record_count = len(records) * multiplier

    print(f"✓ Generated: {output_path}")
    print(f"  Records:       {len(records)} -> {new_record_count}")
    print(f"  Conversations: {original_conv_count} -> {new_conv_count}")
    print(f"  Max chat_id:   {max_chat_id} -> "
          f"{max_chat_id + (multiplier - 1) * chat_id_stride}")
    print(f"  Max timestamp: {max_timestamp:.2f}s -> "
          f"{max_timestamp + (multiplier - 1) * timestamp_stride:.2f}s")
    print()
    print("Use in shell scripts with:")
    print(f"  ./run_mixtral_sync_fixed.sh  --dataset {output_path} "
          f"--num-conv {new_conv_count}")
    print(f"  ./run_mixtral_async_fixed.sh --dataset {output_path} "
          f"--num-conv {new_conv_count}")


if __name__ == "__main__":
    main()
