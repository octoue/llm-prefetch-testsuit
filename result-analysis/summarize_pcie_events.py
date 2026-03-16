#!/usr/bin/env python3
"""
轻量统计 PCIe 事件：各 op 总量、活跃时间、重叠时间、按 rank 差异。

用法:
  python summarize_pcie_events.py results/lite_qps1.2_pcie/pcie_events_0.json
  python summarize_pcie_events.py results/lite_qps1.2_pcie/profiler_output/
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict


def load_events(paths: list[Path]) -> list[dict]:
    events = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        events.extend(data if isinstance(data, list) else [data])
    return events


def overlap_ms(a: dict, b: dict) -> float:
    start_a = a["start_us"]
    end_a = a["end_us"]
    start_b = b["start_us"]
    end_b = b["end_us"]
    return max(0.0, min(end_a, end_b) - max(start_a, start_b)) / 1000.0


def main():
    if len(sys.argv) < 2:
        print("用法: summarize_pcie_events.py <pcie_events.json 或目录>")
        sys.exit(1)

    arg = Path(sys.argv[1])
    if arg.is_dir():
        paths = sorted(arg.glob("pcie_events_*.json"))
    else:
        paths = [arg]

    if not paths:
        print("未找到 pcie_events_*.json")
        sys.exit(1)

    events = load_events(paths)
    if not events:
        print("无事件")
        sys.exit(0)

    for e in events:
        e["end_us"] = e.get("end_us", e["start_us"] + e["duration_ms"] * 1000)

    min_us = min(e["start_us"] for e in events)
    max_us = max(e["end_us"] for e in events)
    window_ms = (max_us - min_us) / 1000.0

    by_op = defaultdict(list)
    for e in events:
        by_op[e["op_type"]].append(e)

    print("=" * 60)
    print("PCIe 事件统计")
    print("=" * 60)
    print(f"文件: {[str(p) for p in paths]}")
    print(f"总事件数: {len(events)}")
    print(f"时间窗口: {window_ms:.1f} ms")
    print()

    print("按 op_type:")
    for op in sorted(by_op.keys()):
        arr = by_op[op]
        total_bytes = sum(e.get("wire_bytes", e["size_bytes"]) for e in arr)
        total_ms = sum(e["duration_ms"] for e in arr)
        bw = (total_bytes / (1024**3)) / (total_ms / 1000) if total_ms > 0 else 0
        by_gpu = Counter(e["gpu_id"] for e in arr)
        print(f"  {op}: count={len(arr)}, total_MB={total_bytes/1024/1024:.1f}, "
              f"active_ms={total_ms:.1f}, bw_GBps={bw:.2f}, gpus={dict(by_gpu)}")

    # 重叠：Prefetch/Restore/Evict vs PP_*
    pp_ops = {"PP_P2P_Send", "PP_P2P_Recv", "PP_TP_AllGather_Reconstruct", "PP_Transfer"}
    pp_events = [e for e in events if e["op_type"] in pp_ops]
    for target in ["Evict", "Restore", "Prefetch"]:
        t_events = by_op.get(target, [])
        if not t_events or not pp_events:
            continue
        total_t = sum(e["end_us"] - e["start_us"] for e in t_events) / 1000.0
        ov = 0.0
        pp_sorted = sorted(pp_events, key=lambda x: x["start_us"])
        j = 0
        for a in sorted(t_events, key=lambda x: x["start_us"]):
            while j < len(pp_sorted) and pp_sorted[j]["end_us"] <= a["start_us"]:
                j += 1
            k = j
            while k < len(pp_sorted) and pp_sorted[k]["start_us"] < a["end_us"]:
                ov += overlap_ms(a, pp_sorted[k])
                k += 1
        ratio = ov / total_t if total_t > 0 else 0
        print(f"\n  {target} vs PP 重叠: {ov:.1f} ms / {total_t:.1f} ms = {ratio:.2%}")

    print("=" * 60)


if __name__ == "__main__":
    main()
