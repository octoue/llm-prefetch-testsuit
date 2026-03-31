#!/usr/bin/env python3
"""
Prefetch A/B 实验 Runner

基于真实 trace 运行 baseline 或 prefetch 模式，逐条落盘到 JSONL。
- baseline: 不发 prefetch，直接发真实请求
- prefetch: 在 scheduled_time - lead_time 发送 history 的 prefetch，在 scheduled_time 发送真实请求

用法:
  python prefetch_ab_runner.py --trace-file qwen_traceA_blksz_16.jsonl --mode baseline --qps 0.5 --output results/baseline.jsonl
  python prefetch_ab_runner.py --trace-file qwen_traceA_blksz_16.jsonl --mode prefetch --qps 0.5 --output results/prefetch.jsonl
"""

import json
import statistics
import time
import asyncio
import argparse
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from openai import AsyncOpenAI
import tiktoken

# 单词池常量（与 test.py 一致）
WORD_POOL_SIZE = 10000
COMMON_WORDS_FOR_POOL = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
]


class PrefetchABRunner:
    def __init__(
        self,
        trace_file: str,
        api_base: str,
        model: str,
        mode: str,
        api_key: str = "dummy",
        prefetch_lead_time: float = 0.2,
        seed: int = 42,
        request_timeout: Optional[float] = None,
    ):
        self.trace_file = trace_file
        self.api_base = api_base
        self.model = model
        self.mode = mode
        self.prefetch_lead_time = prefetch_lead_time  # prefetch 提前量（秒），在 scheduled_time - lead_time 发送
        self.seed = seed
        self.request_timeout = request_timeout  # 仅轻量化测试使用，None 时保持 OpenAI 默认 600s

        # 在 seed 设置后生成单词池，确保两次运行生成完全相同的文本
        random.seed(seed)
        self.word_list = [random.choice(COMMON_WORDS_FOR_POOL) for _ in range(WORD_POOL_SIZE)]

        self.records = []
        self.chat_dict = {}
        self.children_dict = defaultdict(list)
        self.single_turn_conversations = []
        self.multi_turn_conversations = []

        self._load_trace()
        self._analyze_conversations()

        # request_timeout 仅轻量化测试传入，全量测试不传以保持 600s 默认
        client_kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": api_base}
        if request_timeout is not None:
            client_kwargs["timeout"] = request_timeout
        self.client = AsyncOpenAI(**client_kwargs)

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

        self.timeout: Optional[float] = None
        self.test_start_time: Optional[float] = None

        # 进度与统计（用于后台打印）
        self._stats_lock = asyncio.Lock()
        self._completed_count = 0
        self._success_count = 0
        self._ttft_list: List[float] = []
        self._tpot_list: List[float] = []
        self._cached_count = 0
        self._prefetch_cached_list: List[int] = []

    def _load_trace(self):
        with open(self.trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        self.records.append(record)
                        self.chat_dict[record["chat_id"]] = record
                    except json.JSONDecodeError:
                        continue

        for record in self.records:
            parent_id = record["parent_chat_id"]
            if parent_id != -1:
                self.children_dict[parent_id].append(record["chat_id"])

        print(f"加载 trace: {len(self.records)} 条记录")

    def _analyze_conversations(self):
        for record in self.records:
            chat_id = record["chat_id"]
            parent_id = record["parent_chat_id"]
            if parent_id == -1:
                if len(self.children_dict[chat_id]) == 0:
                    self.single_turn_conversations.append(chat_id)
                else:
                    self.multi_turn_conversations.append(chat_id)
        print(f"单轮: {len(self.single_turn_conversations)}, 多轮: {len(self.multi_turn_conversations)}")

    def _get_conversation_chain(self, root_id: int) -> List[Dict]:
        chain = []
        queue = [root_id]
        visited = set()
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            if current_id in self.chat_dict:
                chain.append(self.chat_dict[current_id])
                queue.extend(self.children_dict[current_id])
        chain.sort(key=lambda x: x["turn"])
        return chain

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4

    def _generate_text_with_tokens(
        self,
        target_tokens: int,
        chat_id: Optional[int] = None,
        turn: Optional[int] = None,
        placeholder: bool = False,
    ) -> str:
        """生成具有指定 token 数量的文本。若提供 chat_id/turn，使用确定性子 seed 确保两次运行结果一致。"""
        if target_tokens <= 0:
            return ""
        # 使用确定性子 seed 时，保存并恢复主 RNG 状态
        if chat_id is not None and turn is not None:
            sub_seed = self.seed + chat_id * 1000 + turn + (500000 if placeholder else 0)
            state = random.getstate()
            random.seed(sub_seed)
        try:
            estimated_words = min(int(target_tokens * 1.5), WORD_POOL_SIZE)
            start_idx = random.randint(0, max(0, WORD_POOL_SIZE - estimated_words))
            if start_idx + estimated_words <= WORD_POOL_SIZE:
                text = " ".join(self.word_list[start_idx : start_idx + estimated_words])
            else:
                first = self.word_list[start_idx:]
                remaining = estimated_words - len(first)
                second = self.word_list[:remaining]
                text = " ".join(first + second)
            current = self._count_tokens(text)
            while current < target_tokens and estimated_words < WORD_POOL_SIZE:
                estimated_words += 10
                if start_idx + estimated_words <= WORD_POOL_SIZE:
                    text = " ".join(self.word_list[start_idx : start_idx + estimated_words])
                else:
                    first = self.word_list[start_idx:]
                    remaining = min(estimated_words - len(first), WORD_POOL_SIZE)
                    second = self.word_list[:remaining]
                    text = " ".join(first + second)
                current = self._count_tokens(text)
            return text
        finally:
            if chat_id is not None and turn is not None:
                random.setstate(state)

    def _sample_workload(self, num_multi_turn: int, max_turns: Optional[int] = None) -> List[Tuple[str, List[Dict]]]:
        workload = []
        num_to_take = min(num_multi_turn, len(self.multi_turn_conversations))
        for root_id in self.multi_turn_conversations[:num_to_take]:
            chain = self._get_conversation_chain(root_id)
            if max_turns:
                chain = chain[:max_turns]
            workload.append(("multi", chain))
        return workload

    def _schedule_requests(
        self,
        workload: List[Tuple[str, List[Dict]]],
        qps: float,
        schedule_mode: str = "uniform",
    ) -> Tuple[List, Dict]:
        """返回 (workload, scheduled_map)，scheduled_map[(chat_id,turn)] = 相对时间戳（秒）

        schedule_mode:
          - uniform: 均匀排程，间隔 1/qps
          - scaled-timestamp: 按原始 timestamp 等比例缩放，保留 burst 结构
        """
        all_reqs = []
        for conv_idx, (conv_type, chain) in enumerate(workload):
            for req_idx, record in enumerate(chain):
                all_reqs.append({"conv_idx": conv_idx, "req_idx": req_idx, "record": record})
        all_reqs.sort(key=lambda x: x["record"]["timestamp"])

        if schedule_mode == "scaled-timestamp":
            # 按原始时间戳等比例缩放，保留 burst 结构
            timestamps = [r["record"]["timestamp"] for r in all_reqs]
            t_min, t_max = min(timestamps), max(timestamps)
            orig_dur = t_max - t_min if t_max > t_min else 1.0
            tgt_dur = len(all_reqs) / qps
            scale = tgt_dur / orig_dur
            scheduled_map = {}
            for req in all_reqs:
                r = req["record"]
                rel = (r["timestamp"] - t_min) * scale
                scheduled_map[(r["chat_id"], r["turn"])] = rel
            return workload, scheduled_map

        # uniform
        interval = 1.0 / qps
        scheduled_map = {}
        for i, req in enumerate(all_reqs):
            r = req["record"]
            scheduled_map[(r["chat_id"], r["turn"])] = i * interval
        return workload, scheduled_map

    async def _send_prefetch(
        self, messages: List[Dict]
    ) -> Tuple[float, Optional[str], Optional[int], Optional[int]]:
        """发送 prefetch 请求，返回 (elapsed_sec, error_msg, cached_tokens, prompt_tokens)"""
        start = time.perf_counter()
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"prefetch": True},
            )
            elapsed = time.perf_counter() - start
            cached_tokens = None
            prompt_tokens = None
            if resp.usage:
                prompt_tokens = resp.usage.prompt_tokens
                if hasattr(resp.usage, "prompt_tokens_details") and resp.usage.prompt_tokens_details:
                    details = resp.usage.prompt_tokens_details
                    if hasattr(details, "cached_tokens"):
                        cached_tokens = details.cached_tokens
            return elapsed, None, cached_tokens, prompt_tokens
        except Exception as e:
            return time.perf_counter() - start, str(e), None, None

    async def _send_streaming_request(
        self,
        messages: List[Dict],
        max_tokens: int,
    ) -> Dict:
        start = time.perf_counter()
        first_token_time = None
        usage_info = None
        text = ""
        success = False
        error_msg = None

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"ignore_eos": True},
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    text += chunk.choices[0].delta.content
                if chunk.usage:
                    usage_info = chunk.usage
            success = True
        except Exception as e:
            error_msg = str(e)

        end = time.perf_counter()
        ttft = (first_token_time - start) if first_token_time else (end - start)
        total_time = end - start

        prompt_tokens = usage_info.prompt_tokens if usage_info else 0
        completion_tokens = usage_info.completion_tokens if usage_info else 0
        cached_tokens = None
        if usage_info and hasattr(usage_info, "prompt_tokens_details"):
            details = usage_info.prompt_tokens_details
            if details and hasattr(details, "cached_tokens"):
                cached_tokens = details.cached_tokens

        gen_time = total_time - ttft if total_time > ttft else 0.001
        tokens_per_sec = completion_tokens / gen_time if gen_time > 0 else 0

        # TPOT: Time Per Output Token (秒/token)，与旧 test.py 一致
        if completion_tokens > 1 and first_token_time:
            tpot = (end - first_token_time) / (completion_tokens - 1)
        else:
            tpot = 0.0

        return {
            "ttft": ttft,
            "total_time": total_time,
            "tpot": tpot,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "tokens_per_sec": tokens_per_sec,
            "success": success,
            "error": error_msg,
            "text": text,
        }

    async def _process_conversation(
        self,
        conv_type: str,
        chain: List[Dict],
        scheduled_map: Dict[Tuple[int, int], float],
        output_file,
        output_lock: asyncio.Lock,
        test_start_time: float,
    ):
        messages = []
        for i, record in enumerate(chain):
            if self._is_timeout():
                break

            key = (record["chat_id"], record["turn"])
            scheduled_abs = scheduled_map.get(key, test_start_time)

            # Lead-time 模型：prefetch 在 scheduled_abs - lead_time 发送（仅多轮且有历史）
            prefetch_time_ms = None
            prefetch_cached_tokens = None
            prefetch_prompt_tokens = None
            if self.mode == "prefetch" and conv_type == "multi" and messages:
                prefetch_at = scheduled_abs - self.prefetch_lead_time
                wait_prefetch = prefetch_at - time.time()
                if wait_prefetch > 0:
                    await asyncio.sleep(wait_prefetch)
                if self._is_timeout():
                    break
                elapsed, err, cached, ptokens = await self._send_prefetch(messages.copy())
                prefetch_time_ms = elapsed * 1000
                prefetch_cached_tokens = cached
                prefetch_prompt_tokens = ptokens
                if err:
                    print(f"[Prefetch 失败] chat_id={record['chat_id']} turn={record['turn']}: {err}")
                elif cached is not None and cached > 0 and ptokens is not None:
                    hit_ratio = cached / ptokens if ptokens > 0 else 0
                    source = "GPU_HIT" if elapsed < 0.1 else "CPU_LOAD"
                    print(
                        f"[Prefetch {source}] chat_id={record['chat_id']} turn={record['turn']}: "
                        f"cached={cached}/{ptokens} ({hit_ratio:.0%}), elapsed={elapsed*1000:.0f}ms"
                    )

            # 等待到该轮次的预定发送时间
            wait_time = scheduled_abs - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            if self._is_timeout():
                break

            history_tokens = self._count_tokens("\n".join(f"{m['role']}: {m['content']}" for m in messages)) if messages else 0
            new_user_tokens = max(10, record["input_length"] - history_tokens)
            user_msg = self._generate_text_with_tokens(
                new_user_tokens, chat_id=record["chat_id"], turn=record["turn"]
            )
            messages.append({"role": "user", "content": user_msg})

            result = await self._send_streaming_request(messages, record["output_length"])

            if result["success"] and result.get("text"):
                messages.append({"role": "assistant", "content": result["text"]})
            else:
                messages.append({
                    "role": "assistant",
                    "content": self._generate_text_with_tokens(
                        record["output_length"],
                        chat_id=record["chat_id"],
                        turn=record["turn"],
                        placeholder=True,
                    ),
                })

            log_row = {
                "chat_id": record["chat_id"],
                "turn": record["turn"],
                "mode": self.mode,
                "is_multi_turn": conv_type == "multi",
                "ttft_ms": result["ttft"] * 1000,
                "tpot_ms": result["tpot"] * 1000,
                "total_time_ms": result["total_time"] * 1000,
                "tokens_per_sec": result["tokens_per_sec"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "cached_tokens": result["cached_tokens"],
                "prefetch_time_ms": prefetch_time_ms,
                "prefetch_elapsed_ms": prefetch_time_ms,
                "prefetch_cached_tokens": prefetch_cached_tokens,
                "prefetch_prompt_tokens": prefetch_prompt_tokens,
                "history_tokens": history_tokens if conv_type == "multi" else None,
                "success": result["success"],
                "error": result["error"],
                "timestamp": time.time(),
            }
            async with output_lock:
                output_file.write(json.dumps(log_row, ensure_ascii=False) + "\n")
                output_file.flush()

            # 更新进度统计
            async with self._stats_lock:
                self._completed_count += 1
                if result["success"]:
                    self._success_count += 1
                    self._ttft_list.append(result["ttft"] * 1000)
                    self._tpot_list.append(result["tpot"] * 1000)
                if result.get("cached_tokens") and result["cached_tokens"] > 0:
                    self._cached_count += 1
                if prefetch_cached_tokens is not None and prefetch_cached_tokens > 0:
                    self._prefetch_cached_list.append(prefetch_cached_tokens)

            status = "✓" if result["success"] else "✗"
            cached_str = f", cached={result['cached_tokens']}" if result["cached_tokens"] is not None else ""
            print(f"[{status}] {record['chat_id']}_t{record['turn']} TTFT={result['ttft']*1000:.0f}ms{cached_str}")

    def _is_timeout(self) -> bool:
        if self.timeout is None or self.test_start_time is None:
            return False
        return time.time() - self.test_start_time >= self.timeout

    async def run(
        self,
        num_multi_turn: int,
        qps: float,
        output: str,
        max_turns: Optional[int] = None,
        timeout: Optional[float] = None,
        schedule_mode: str = "uniform",
    ):
        workload = self._sample_workload(num_multi_turn, max_turns)
        if not workload:
            print("无 workload")
            return

        scheduled_workload, scheduled_map_rel = self._schedule_requests(
            workload, qps, schedule_mode=schedule_mode
        )

        self.test_start_time = time.time()
        self.timeout = timeout
        base_time = self.test_start_time
        scheduled_map = {
            k: base_time + v for k, v in scheduled_map_rel.items()
        }

        total_requests = sum(len(c) for _, c in workload)
        print(f"\n模式={self.mode}, QPS={qps}, 总请求={total_requests}")
        print("-" * 60)

        async def _progress_reporter() -> None:
            """每 10 秒打印进度摘要"""
            while True:
                await asyncio.sleep(10)
                async with self._stats_lock:
                    c = self._completed_count
                    s = self._success_count
                    f = c - s
                    elapsed = time.time() - base_time
                if c >= total_requests:
                    break
                pct = 100 * c / total_requests if total_requests else 0
                eta = (elapsed / c * (total_requests - c)) if c > 0 and total_requests else 0
                print(f"[Progress] {c}/{total_requests} requests ({pct:.1f}%), success={s}, failed={f}, elapsed {elapsed:.0f}s, ETA {eta:.0f}s")
                if c > 0 and self._ttft_list:
                    ttft_mean = statistics.mean(self._ttft_list)
                    ttft_p50 = statistics.median(self._ttft_list)
                    tpot_mean = statistics.mean(self._tpot_list) if self._tpot_list else 0
                    hit = 100 * self._cached_count / c
                    print(f"[Stats] TTFT mean={ttft_mean:.0f}ms p50={ttft_p50:.0f}ms | TPOT mean={tpot_mean:.1f}ms | Cache hit={hit:.1f}%")

        output_lock = asyncio.Lock()
        progress_task: Optional[asyncio.Task] = None
        try:
            with open(output, "w", encoding="utf-8") as out_f:
                tasks = []
                for conv_type, chain in scheduled_workload:
                    task = asyncio.create_task(
                        self._process_conversation(conv_type, chain, scheduled_map, out_f, output_lock, base_time)
                    )
                    tasks.append(task)
                progress_task = asyncio.create_task(_progress_reporter())
                await asyncio.gather(*tasks)
        finally:
            if progress_task and not progress_task.done():
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

        # 最终汇总
        async with self._stats_lock:
            c = self._completed_count
            s = self._success_count
            elapsed = time.time() - base_time
        print(f"\n[完成] {c} 请求, 成功={s}, 失败={c-s}, 耗时 {elapsed:.1f}s")
        if self._ttft_list:
            print(f"[TTFT] mean={statistics.mean(self._ttft_list):.0f}ms, p50={statistics.median(self._ttft_list):.0f}ms")
        if self._tpot_list:
            print(f"[TPOT] mean={statistics.mean(self._tpot_list):.1f}ms")
        print(f"\n结果已写入: {output}")


async def main():
    parser = argparse.ArgumentParser(description="Prefetch A/B Runner")
    parser.add_argument("--trace-file", required=True, help="Trace JSONL 路径")
    parser.add_argument("--mode", required=True, choices=["baseline", "prefetch"])
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="/lpai/models/Qwen__Qwen3-8B/25-07-26-0349")
    parser.add_argument("--qps", type=float, default=0.5)
    parser.add_argument("--num-multi-turn", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--prefetch-lead-time", type=float, default=0.2, help="Prefetch 提前量（秒），在 scheduled_time - lead_time 发送 prefetch")
    parser.add_argument("--schedule-mode", choices=["uniform", "scaled-timestamp"], default="uniform",
                        help="uniform=均匀排程; scaled-timestamp=按原始时间戳缩放保留 burst")
    parser.add_argument("--timeout", type=float, default=None, help="测试总超时（秒），超时后不再发送新请求。全量测试不传以跑完所有请求")
    parser.add_argument("--request-timeout", type=float, default=None, help="单请求 HTTP 超时（秒）。仅轻量化测试传入（如 120），全量测试不传以保持 600s 默认")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，确保两次运行生成完全相同的对话文本")
    args = parser.parse_args()

    runner = PrefetchABRunner(
        trace_file=args.trace_file,
        api_base=args.api_base,
        model=args.model,
        mode=args.mode,
        prefetch_lead_time=args.prefetch_lead_time,
        seed=args.seed,
        request_timeout=args.request_timeout,
    )
    await runner.run(
        num_multi_turn=args.num_multi_turn,
        qps=args.qps,
        output=args.output,
        max_turns=args.max_turns,
        timeout=args.timeout,
        schedule_mode=args.schedule_mode,
    )


if __name__ == "__main__":
    asyncio.run(main())
