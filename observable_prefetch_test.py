#!/usr/bin/env python3
"""
可观测的 Prefetch 集成测试脚本

验证两件事：
1. CPU→GPU KV Block 回迁行为正确性（prefetch 从 CPU offload 加载 KV 到 GPU）
2. Prefetch 对 TTFT 的实际效果（有 prefetch vs 无 prefetch 的 TTFT 对比）

前提条件：
- vLLM 需以 --enable-prefix-caching --kv-offloading-size N 启动
- 需要 GPU 环境

用法:
  python observable_prefetch_test.py [--base-url URL] [--model MODEL]
"""

import argparse
import asyncio
import time
from openai import AsyncOpenAI


# 用于构造对话历史的固定内容
HISTORY_A = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": (
            "Please explain the theory of relativity in detail, "
            "covering both special and general relativity."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The theory of relativity, proposed by Albert Einstein, "
            "is divided into two parts: special relativity (1905) and "
            "general relativity (1915). Special relativity deals with "
            "objects moving at constant speeds, particularly near the "
            "speed of light. Its key postulates are that the laws of "
            "physics are the same in all inertial frames."
        ),
    },
]

# 用于填满 GPU cache 的其他对话（不同 prefix）
FILLER_PROMPTS = [
    f"Tell me about topic {i} in great detail with many paragraphs."
    for i in range(100)
]


async def measure_ttft(client: AsyncOpenAI, model: str, messages: list) -> tuple[float, int, int | None]:
    """发送请求并测量 TTFT，返回 (ttft_sec, prompt_tokens, cached_tokens)。"""
    start = time.perf_counter()
    first_token_time: float | None = None
    usage = None

    stream_obj = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=32,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream_obj:
        if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_time = time.perf_counter() - start
        if chunk.usage:
            usage = chunk.usage
    ttft = first_token_time if first_token_time is not None else time.perf_counter() - start
    if usage:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = details.cached_tokens if details and hasattr(details, "cached_tokens") else None
        return ttft, usage.prompt_tokens, cached
    return ttft, 0, None


async def send_prefetch(client: AsyncOpenAI, model: str, messages: list) -> dict:
    """发送 prefetch 请求。"""
    start = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"prefetch": True},
    )
    elapsed = time.perf_counter() - start
    usage = resp.usage
    details = getattr(usage, "prompt_tokens_details", None)
    cached = details.cached_tokens if details and hasattr(details, "cached_tokens") else None
    return {
        "elapsed": elapsed,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": cached,
    }


async def send_normal_request(client: AsyncOpenAI, model: str, messages: list) -> dict:
    """发送普通请求，填充 KV cache。"""
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=8,
        stream=False,
    )
    usage = resp.usage
    details = getattr(usage, "prompt_tokens_details", None)
    cached = details.cached_tokens if details and hasattr(details, "cached_tokens") else None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": cached,
    }


async def run_experiment_1(client: AsyncOpenAI, model: str) -> None:
    """
    实验 1: 验证 CPU→GPU KV Block 回迁

    步骤:
    1. 发送对话 A 的正常请求，KV 被计算并存于 GPU
    2. 发送大量其他对话填满 GPU KV cache → 对话 A 的 KV 被挤出到 CPU
    3. 发送对话 A 的 prefetch 请求 → 应触发 CPU→GPU load
    4. 发送对话 A 的真实请求 → 验证 cached_tokens > 0
    """
    print("\n" + "=" * 60)
    print("=== Experiment 1: CPU→GPU KV Block Reclaim ===")
    print("=" * 60)

    # Step 1: 正常请求 A
    print("\n[Step 1] Sending normal request for conversation A...")
    ttft1, pt1, _ = await measure_ttft(client, model, HISTORY_A)
    print(f"  prompt_tokens={pt1}, TTFT={ttft1*1000:.0f}ms")

    # Step 2: 填满 GPU cache
    print("\n[Step 2] Filling GPU cache with other conversations...")
    num_fillers = 30
    for i in range(num_fillers):
        msgs = [{"role": "user", "content": FILLER_PROMPTS[i]}]
        await send_normal_request(client, model, msgs)
        if (i + 1) % 10 == 0:
            print(f"  Sent {i + 1}/{num_fillers} filler requests...")
    print("  Done.")

    # Step 3: Prefetch 对话 A
    print("\n[Step 3] Sending prefetch request for conversation A...")
    prefetch_result = await send_prefetch(client, model, HISTORY_A)
    print(
        f"  status=ok, prompt_tokens={prefetch_result['prompt_tokens']}, "
        f"completion_tokens={prefetch_result['completion_tokens']}, "
        f"elapsed={prefetch_result['elapsed']*1000:.0f}ms"
    )

    # Step 4: 真实请求 A（追加新 user message）
    print("\n[Step 4] Sending real request for conversation A (with new message)...")
    real_messages = HISTORY_A + [
        {"role": "user", "content": "Can you summarize that in one sentence?"}
    ]
    ttft2, pt2, cached = await measure_ttft(client, model, real_messages)
    print(f"  prompt_tokens={pt2}, cached_tokens={cached}, TTFT={ttft2*1000:.0f}ms")

    if cached is not None and cached > 0:
        print(f"\n✓ CPU→GPU reclaim verified: cached_tokens={cached} > 0")
    else:
        print(
            f"\n⚠ cached_tokens={cached} (expected > 0). "
            "Ensure vLLM is started with --kv-offloading-size and --enable-prefix-caching."
        )


async def run_experiment_2(client: AsyncOpenAI, model: str) -> None:
    """
    实验 2: TTFT 效果对比

    步骤:
    1. 构造场景使 KV 被挤到 CPU
    2. 对照组: 不发 prefetch，直接发真实请求 → 记录 TTFT
    3. 实验组: 先发 prefetch，再发真实请求 → 记录 TTFT
    4. 对比
    """
    print("\n" + "=" * 60)
    print("=== Experiment 2: TTFT Comparison (With vs Without Prefetch) ===")
    print("=" * 60)

    # 先填满 cache
    print("\n[Setup] Filling GPU cache...")
    for i in range(25):
        msgs = [{"role": "user", "content": FILLER_PROMPTS[i]}]
        await send_normal_request(client, model, msgs)
    print("  Done.")

    real_messages = HISTORY_A + [
        {"role": "user", "content": "What is E=mc^2?"}
    ]

    # 对照组: 无 prefetch
    print("\n[Control] Sending real request WITHOUT prefetch...")
    ttft_control, _, _ = await measure_ttft(client, model, real_messages)
    print(f"  TTFT = {ttft_control*1000:.0f}ms")

    # 再次填满 cache（确保下次请求时 KV 在 CPU）
    for i in range(25, 50):
        msgs = [{"role": "user", "content": FILLER_PROMPTS[i]}]
        await send_normal_request(client, model, msgs)

    # 实验组: 有 prefetch
    print("\n[Treatment] Sending prefetch, then real request...")
    await send_prefetch(client, model, HISTORY_A)
    ttft_treatment, _, cached = await measure_ttft(client, model, real_messages)
    print(f"  TTFT = {ttft_treatment*1000:.0f}ms, cached_tokens={cached}")

    # 对比
    if ttft_control > 0:
        reduction = (1 - ttft_treatment / ttft_control) * 100
        print(f"\n→ TTFT reduced by {reduction:.1f}% (with prefetch)")
    else:
        print("\n→ Could not compute reduction (control TTFT was 0)")


async def main():
    parser = argparse.ArgumentParser(
        description="Observable prefetch integration tests"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM API base URL",
    )
    parser.add_argument(
        "--model",
        default="/lpai/models/Qwen__Qwen3-8B/25-07-26-0349",
        help="Model name or path",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only experiment 1 or 2 (default: run both)",
    )
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")

    print("Prefetch Observable Tests")
    print(f"  Base URL: {args.base_url}")
    print(f"  Model: {args.model}")

    try:
        if args.experiment is None or args.experiment == 1:
            await run_experiment_1(client, args.model)
        if args.experiment is None or args.experiment == 2:
            await run_experiment_2(client, args.model)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
