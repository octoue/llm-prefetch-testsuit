#!/usr/bin/env python3
"""
Prefetch A/B 实验 Runner

基于真实 trace 运行 baseline 或 prefetch 模式，逐条落盘到 JSONL。
- baseline: 不发 prefetch，直接发真实请求
- prefetch: 先发 history 的 prefetch=True，等待 thinking time，再发真实请求

用法:
  python prefetch_ab_runner.py --trace-file qwen_traceA_blksz_16.jsonl --mode baseline --qps 0.5 --output results/baseline.jsonl
  python prefetch_ab_runner.py --trace-file qwen_traceA_blksz_16.jsonl --mode prefetch --qps 0.5 --output results/prefetch.jsonl
"""

import json
import time
import asyncio
import argparse
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from openai import AsyncOpenAI
import tiktoken

# 全局单词池（与 test.py 一致）
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

GLOBAL_WORD_LIST = [random.choice(COMMON_WORDS_FOR_POOL) for _ in range(WORD_POOL_SIZE)]


class PrefetchABRunner:
    def __init__(
        self,
        trace_file: str,
        api_base: str,
        model: str,
        mode: str,
        api_key: str = "dummy",
        thinking_time: Optional[float] = None,
    ):
        self.trace_file = trace_file
        self.api_base = api_base
        self.model = model
        self.mode = mode
        self.thinking_time = thinking_time  # 固定 thinking time（秒），None 则从 trace 推导

        self.records = []
        self.chat_dict = {}
        self.children_dict = defaultdict(list)
        self.single_turn_conversations = []
        self.multi_turn_conversations = []

        self._load_trace()
        self._analyze_conversations()

        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

        self.timeout: Optional[float] = None
        self.test_start_time: Optional[float] = None

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

    def _generate_text_with_tokens(self, target_tokens: int) -> str:
        if target_tokens <= 0:
            return ""
        estimated_words = min(int(target_tokens * 1.5), WORD_POOL_SIZE)
        start_idx = random.randint(0, max(0, WORD_POOL_SIZE - estimated_words))
        if start_idx + estimated_words <= WORD_POOL_SIZE:
            text = " ".join(GLOBAL_WORD_LIST[start_idx : start_idx + estimated_words])
        else:
            first = GLOBAL_WORD_LIST[start_idx:]
            remaining = estimated_words - len(first)
            second = GLOBAL_WORD_LIST[:remaining]
            text = " ".join(first + second)
        current = self._count_tokens(text)
        while current < target_tokens and estimated_words < WORD_POOL_SIZE:
            estimated_words += 10
            if start_idx + estimated_words <= WORD_POOL_SIZE:
                text = " ".join(GLOBAL_WORD_LIST[start_idx : start_idx + estimated_words])
            else:
                first = GLOBAL_WORD_LIST[start_idx:]
                remaining = min(estimated_words - len(first), WORD_POOL_SIZE)
                second = GLOBAL_WORD_LIST[:remaining]
                text = " ".join(first + second)
            current = self._count_tokens(text)
        return text

    def _sample_workload(self, num_multi_turn: int, max_turns: Optional[int] = None) -> List[Tuple[str, List[Dict]]]:
        workload = []
        num_to_take = min(num_multi_turn, len(self.multi_turn_conversations))
        for root_id in self.multi_turn_conversations[:num_to_take]:
            chain = self._get_conversation_chain(root_id)
            if max_turns:
                chain = chain[:max_turns]
            workload.append(("multi", chain))
        return workload

    def _schedule_requests(self, workload: List[Tuple[str, List[Dict]]], qps: float) -> Tuple[List, Dict]:
        """返回 (workload, scheduled_map)，scheduled_map[(chat_id,turn)] = 绝对时间戳"""
        all_reqs = []
        for conv_idx, (conv_type, chain) in enumerate(workload):
            for req_idx, record in enumerate(chain):
                all_reqs.append({"conv_idx": conv_idx, "req_idx": req_idx, "record": record})
        all_reqs.sort(key=lambda x: x["record"]["timestamp"])
        interval = 1.0 / qps
        scheduled_map = {}
        for i, req in enumerate(all_reqs):
            r = req["record"]
            scheduled_map[(r["chat_id"], r["turn"])] = i * interval  # 相对时间
        return workload, scheduled_map

    async def _send_prefetch(self, messages: List[Dict]) -> Tuple[float, Optional[str]]:
        """发送 prefetch 请求，返回 (elapsed_sec, error_msg)"""
        start = time.perf_counter()
        try:
            await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"prefetch": True},
            )
            return time.perf_counter() - start, None
        except Exception as e:
            return time.perf_counter() - start, str(e)

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

        return {
            "ttft": ttft,
            "total_time": total_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "tokens_per_sec": tokens_per_sec,
            "success": success,
            "error": error_msg,
            "text": text,
        }

    def _get_thinking_time(self, chain: List[Dict], req_idx: int) -> float:
        if self.thinking_time is not None:
            return self.thinking_time
        if req_idx == 0:
            return 0.0
        prev_ts = chain[req_idx - 1]["timestamp"]
        curr_ts = chain[req_idx]["timestamp"]
        return max(0.0, curr_ts - prev_ts)

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
            wait_time = scheduled_abs - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            if self._is_timeout():
                break

            history_tokens = self._count_tokens("\n".join(f"{m['role']}: {m['content']}" for m in messages)) if messages else 0
            new_user_tokens = max(10, record["input_length"] - history_tokens)
            user_msg = self._generate_text_with_tokens(new_user_tokens)
            messages.append({"role": "user", "content": user_msg})

            prefetch_time_ms = None
            if self.mode == "prefetch" and conv_type == "multi" and len(messages) > 1:
                history_messages = messages[:-1]
                thinking = self._get_thinking_time(chain, i)
                if thinking > 0:
                    await asyncio.sleep(thinking)
                elapsed, err = await self._send_prefetch(history_messages)
                prefetch_time_ms = elapsed * 1000
                if err:
                    print(f"[Prefetch 失败] chat_id={record['chat_id']} turn={record['turn']}: {err}")

            result = await self._send_streaming_request(messages, record["output_length"])

            if result["success"] and result.get("text"):
                messages.append({"role": "assistant", "content": result["text"]})
            else:
                messages.append({"role": "assistant", "content": self._generate_text_with_tokens(record["output_length"])})

            log_row = {
                "chat_id": record["chat_id"],
                "turn": record["turn"],
                "mode": self.mode,
                "is_multi_turn": conv_type == "multi",
                "ttft_ms": result["ttft"] * 1000,
                "total_time_ms": result["total_time"] * 1000,
                "tokens_per_sec": result["tokens_per_sec"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "cached_tokens": result["cached_tokens"],
                "prefetch_time_ms": prefetch_time_ms,
                "success": result["success"],
                "error": result["error"],
                "timestamp": time.time(),
            }
            async with output_lock:
                output_file.write(json.dumps(log_row, ensure_ascii=False) + "\n")
                output_file.flush()

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
    ):
        workload = self._sample_workload(num_multi_turn, max_turns)
        if not workload:
            print("无 workload")
            return

        scheduled_workload, scheduled_map_rel = self._schedule_requests(workload, qps)

        self.test_start_time = time.time()
        self.timeout = timeout
        base_time = self.test_start_time
        scheduled_map = {
            k: base_time + v for k, v in scheduled_map_rel.items()
        }

        total_requests = sum(len(c) for _, c in workload)
        print(f"\n模式={self.mode}, QPS={qps}, 总请求={total_requests}")
        print("-" * 60)

        output_lock = asyncio.Lock()
        with open(output, "w", encoding="utf-8") as out_f:
            tasks = []
            for conv_type, chain in scheduled_workload:
                task = asyncio.create_task(
                    self._process_conversation(conv_type, chain, scheduled_map, out_f, output_lock, base_time)
                )
                tasks.append(task)
            await asyncio.gather(*tasks)

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
    parser.add_argument("--thinking-time", type=float, default=None, help="固定 thinking time（秒），默认从 trace 推导")
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    runner = PrefetchABRunner(
        trace_file=args.trace_file,
        api_base=args.api_base,
        model=args.model,
        mode=args.mode,
        thinking_time=args.thinking_time,
    )
    await runner.run(
        num_multi_turn=args.num_multi_turn,
        qps=args.qps,
        output=args.output,
        max_turns=args.max_turns,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    asyncio.run(main())
