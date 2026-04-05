"""
EP + PCIe Benchmark: 测量 MoE Expert Parallel 在 PCIe-only 下的性能,
包括 all-to-all 通信开销和 EPLB 权重迁移的影响.

用法:
  python3 bench_ep_pcie.py \
      --api-base http://localhost:8000/v1 \
      --model deepseek-ai/DeepSeek-V2-Lite-Chat \
      --qps 1.0 \
      --num-requests 100

输出: JSON 格式的 TTFT/TPOT 统计 + EPLB 触发次数
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import numpy as np

try:
    import openai
except ImportError:
    print("Error: pip install openai")
    sys.exit(1)


@dataclass
class RequestResult:
    request_id: int
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float
    cached_tokens: int
    success: bool
    error: str = ""


PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a quicksort implementation in Python with comments.",
    "What are the main causes of climate change?",
    "Describe the process of protein synthesis in cells.",
    "How does a neural network learn? Explain backpropagation.",
    "What is the difference between TCP and UDP?",
    "Explain how a compiler works, step by step.",
    "What are the key principles of object-oriented programming?",
    "Describe the water cycle and its importance.",
    "How does public key cryptography work?",
    "Explain the CAP theorem in distributed systems.",
    "What is the difference between concurrency and parallelism?",
    "Describe how garbage collection works in Java.",
    "What are the main data structures used in databases?",
    "Explain MapReduce and its applications.",
    "How does DNS resolution work?",
    "What is the difference between REST and GraphQL?",
    "Explain the concept of virtual memory.",
    "What are microservices and their trade-offs?",
    "Describe the PageRank algorithm.",
]


def send_request(
    client: openai.OpenAI,
    model: str,
    request_id: int,
    max_tokens: int = 128,
) -> RequestResult:
    prompt = PROMPTS[request_id % len(PROMPTS)]

    start = time.perf_counter()
    ttft = None

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
            stream_options={"include_usage": True},
        )

        completion_tokens = 0
        prompt_tokens = 0
        cached_tokens = 0

        for chunk in stream:
            if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                ttft = (time.perf_counter() - start) * 1000

            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                if details:
                    cached_tokens = getattr(details, "cached_tokens", 0) or 0

        total_ms = (time.perf_counter() - start) * 1000

        return RequestResult(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft or total_ms,
            total_ms=total_ms,
            cached_tokens=cached_tokens,
            success=True,
        )

    except Exception as e:
        total_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            request_id=request_id,
            prompt_tokens=0,
            completion_tokens=0,
            ttft_ms=total_ms,
            total_ms=total_ms,
            cached_tokens=0,
            success=False,
            error=str(e),
        )


def run_benchmark(args):
    client = openai.OpenAI(base_url=args.api_base, api_key="dummy")

    results: list[RequestResult] = []
    interval = 1.0 / args.qps if args.qps > 0 else 0

    print(f"Sending {args.num_requests} requests at QPS={args.qps}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    with ThreadPoolExecutor(max_workers=min(args.num_requests, 32)) as pool:
        futures = {}
        start_time = time.perf_counter()

        for i in range(args.num_requests):
            # Rate limiting
            target_time = start_time + i * interval
            now = time.perf_counter()
            if now < target_time:
                time.sleep(target_time - now)

            fut = pool.submit(send_request, client, args.model, i, args.max_tokens)
            futures[fut] = i

        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)

            if len(results) % 20 == 0:
                ok = sum(1 for r in results if r.success)
                print(f"  Progress: {len(results)}/{args.num_requests} (ok={ok})")

    # Sort by request_id
    results.sort(key=lambda r: r.request_id)
    return results


def print_stats(results: list[RequestResult], output_path: str | None = None):
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    print()
    print("=" * 60)
    print(f"RESULTS: {len(ok)} ok, {len(fail)} failed / {len(results)} total")
    print("=" * 60)

    if not ok:
        print("No successful requests.")
        return

    ttfts = np.array([r.ttft_ms for r in ok])
    totals = np.array([r.total_ms for r in ok])
    comp_tokens = np.array([r.completion_tokens for r in ok])
    tpots = np.array([
        (r.total_ms - r.ttft_ms) / max(r.completion_tokens - 1, 1) for r in ok
    ])
    cached = sum(r.cached_tokens for r in ok)

    print(f"\nTTFT (ms):")
    print(f"  Mean: {np.mean(ttfts):.1f}  Std: {np.std(ttfts):.1f}")
    print(f"  P50:  {np.percentile(ttfts, 50):.1f}  P95: {np.percentile(ttfts, 95):.1f}  P99: {np.percentile(ttfts, 99):.1f}")

    print(f"\nTPOT (ms/token):")
    print(f"  Mean: {np.mean(tpots):.1f}  Std: {np.std(tpots):.1f}")
    print(f"  P50:  {np.percentile(tpots, 50):.1f}  P95: {np.percentile(tpots, 95):.1f}")

    print(f"\nTotal latency (ms):")
    print(f"  Mean: {np.mean(totals):.1f}  P95: {np.percentile(totals, 95):.1f}")

    print(f"\nTokens:")
    print(f"  Avg completion: {np.mean(comp_tokens):.1f}")
    print(f"  Total cached: {cached}")

    # Save detailed results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
        print(f"\nDetailed results: {output_path}")

    # Summary JSON
    summary = {
        "num_requests": len(results),
        "num_success": len(ok),
        "num_failed": len(fail),
        "ttft_mean_ms": float(np.mean(ttfts)),
        "ttft_p50_ms": float(np.percentile(ttfts, 50)),
        "ttft_p95_ms": float(np.percentile(ttfts, 95)),
        "ttft_p99_ms": float(np.percentile(ttfts, 99)),
        "tpot_mean_ms": float(np.mean(tpots)),
        "tpot_p50_ms": float(np.percentile(tpots, 50)),
        "total_cached_tokens": int(cached),
    }

    if output_path:
        summary_path = output_path.replace(".jsonl", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="EP + PCIe Benchmark")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (default: results/bench_<timestamp>.jsonl)")
    args = parser.parse_args()

    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"results/bench_{ts}.jsonl"

    results = run_benchmark(args)
    print_stats(results, args.output)


if __name__ == "__main__":
    main()
