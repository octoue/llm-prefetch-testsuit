#!/usr/bin/env python3
"""写入实验元数据 JSON，便于对比远端实验结果。"""
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--qps", type=float, default=0)
    p.add_argument("--num-conv", type=int, default=0)
    p.add_argument("--schedule-mode", default="uniform")
    p.add_argument("--lead-time", type=float, default=0)
    p.add_argument("--trace", default="")
    p.add_argument("--sampling-mode", default="default")
    p.add_argument("--max-input-length", default="3000")
    args = p.parse_args()
    meta = {
        "qps": args.qps,
        "num_conv": args.num_conv,
        "schedule_mode": args.schedule_mode,
        "prefetch_lead_time": args.lead_time,
        "trace": args.trace,
        "sampling_mode": args.sampling_mode,
        "max_input_length": args.max_input_length,
    }
    with open(args.output, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
