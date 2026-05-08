#!/usr/bin/env python3
"""Stress runner for extreme prefetch failure scenarios.

Three workloads are supported:

  S1 malicious flood        Continuous prefetch-only requests at a fixed QPS,
                            never followed by real inference. Tests admission
                            control's ability to absorb pure attack traffic.
  S2 misclick burst         For each window of a real-trace conversation,
                            send N prefetch requests, only the last one
                            followed by the actual inference call. Tests how
                            quota + TTL reclaim absorb dense legitimate-looking
                            bursts.
  S3 mixed background       S1 attack traffic + a real trace running in
                            parallel. Used to measure isolation of active
                            users from the attacker.

For all modes the runner periodically scrapes ``/metrics`` to capture
``vllm:prefetch_*_total`` counters at 1 Hz and writes a JSON time series
alongside the per-request JSONL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore


WORD_POOL_SIZE = 10000
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
]


PREFETCH_METRIC_NAMES = (
    "vllm:prefetch_gpu_hits_total",
    "vllm:prefetch_cpu_hits_total",
    "vllm:prefetch_no_hits_total",
    "vllm:prefetch_deferred_total",
    "vllm:prefetch_expired_total",
)


def _build_word_list(seed: int) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(COMMON_WORDS) for _ in range(WORD_POOL_SIZE)]


def _gen_text(word_list: list[str], target_tokens: int, tokenizer) -> str:
    """Generate a text whose tiktoken length is >= target_tokens."""
    if target_tokens <= 0:
        return ""
    estimated = min(int(target_tokens * 1.5), WORD_POOL_SIZE)
    rng = random.Random(target_tokens)
    start = rng.randint(0, max(0, WORD_POOL_SIZE - estimated))
    text = " ".join(word_list[start : start + estimated])
    if tokenizer is None:
        return text
    cur = len(tokenizer.encode(text))
    while cur < target_tokens and estimated < WORD_POOL_SIZE:
        estimated += 32
        end = min(start + estimated, WORD_POOL_SIZE)
        text = " ".join(word_list[start:end])
        cur = len(tokenizer.encode(text))
    return text


def _load_trace(path: str, max_records: int | None = None) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if max_records is not None and len(out) > max_records:
        out = out[:max_records]
    return out


def _build_chains(records: list[dict]) -> list[list[dict]]:
    """Materialize multi-turn conversation chains, oldest turn first."""
    by_id = {r["chat_id"]: r for r in records}
    children: dict[int, list[int]] = defaultdict(list)
    roots: list[int] = []
    for r in records:
        if r["parent_chat_id"] == -1:
            roots.append(r["chat_id"])
        else:
            children[r["parent_chat_id"]].append(r["chat_id"])
    chains: list[list[dict]] = []
    for root in roots:
        # BFS yields turns in arrival order; sort by turn for safety.
        chain, queue, visited = [], [root], set()
        while queue:
            cid = queue.pop(0)
            if cid in visited:
                continue
            visited.add(cid)
            if cid in by_id:
                chain.append(by_id[cid])
                queue.extend(children[cid])
        chain.sort(key=lambda x: x["turn"])
        if len(chain) >= 2:
            chains.append(chain)
    return chains


class MetricsScraper:
    """Polls /metrics at a fixed period and records prefetch counter deltas."""

    def __init__(self, base_url: str, period_sec: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.period = period_sec
        self.samples: list[dict] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._client: httpx.AsyncClient | None = None
        self._t0: float | None = None

    @staticmethod
    def _parse_prom(text: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name not in PREFETCH_METRIC_NAMES:
                continue
            try:
                value = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            out[name] = out.get(name, 0.0) + value
        return out

    async def _loop(self) -> None:
        assert self._client is not None
        while not self._stop.is_set():
            t = time.time()
            try:
                resp = await self._client.get(f"{self.base_url}/metrics")
                if resp.status_code == 200:
                    sample = self._parse_prom(resp.text)
                    sample["t_rel_sec"] = round(t - (self._t0 or t), 3)
                    self.samples.append(sample)
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.period)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        if self._task is not None:
            return
        self._client = httpx.AsyncClient(timeout=2.0)
        self._t0 = time.time()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await self._task
            except Exception:
                pass
        if self._client:
            await self._client.aclose()


class StressRunner:
    def __init__(
        self,
        api_base: str,
        model: str,
        word_list: list[str],
        tokenizer,
        out_jsonl,
        out_lock: asyncio.Lock,
    ):
        self.api_base = api_base
        self.model = model
        self.word_list = word_list
        self.tokenizer = tokenizer
        self.out_jsonl = out_jsonl
        self.out_lock = out_lock
        self.client = AsyncOpenAI(api_key="dummy", base_url=api_base, timeout=30.0)

        self.prefetch_sent = 0
        self.prefetch_ack = 0
        self.prefetch_rejected = 0
        self.real_sent = 0
        self.real_success = 0
        self.real_ttft_ms: list[float] = []
        self.real_tpot_ms: list[float] = []

    async def _record(self, row: dict) -> None:
        async with self.out_lock:
            self.out_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.out_jsonl.flush()

    async def send_prefetch(
        self, messages: list[dict], tag: str, chat_id: int, turn: int
    ) -> None:
        self.prefetch_sent += 1
        start = time.perf_counter()
        cached = None
        prompt = None
        err = None
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"prefetch": True},
            )
            if resp.usage:
                prompt = resp.usage.prompt_tokens
                details = getattr(resp.usage, "prompt_tokens_details", None)
                if details and getattr(details, "cached_tokens", None) is not None:
                    cached = details.cached_tokens
            self.prefetch_ack += 1
            # In current vLLM, a "deferred" or "no-hit" prefetch still returns
            # 200 with cached_tokens=0/None. We log the raw signal; final
            # classification is done against the Prometheus delta.
            if cached is None or cached == 0:
                self.prefetch_rejected += 1
        except Exception as e:
            err = str(e)
        elapsed_ms = (time.perf_counter() - start) * 1000
        await self._record(
            {
                "kind": "prefetch",
                "tag": tag,
                "chat_id": chat_id,
                "turn": turn,
                "elapsed_ms": elapsed_ms,
                "cached_tokens": cached,
                "prompt_tokens": prompt,
                "error": err,
                "ts": time.time(),
            }
        )

    async def send_real(
        self,
        messages: list[dict],
        max_tokens: int,
        tag: str,
        chat_id: int,
        turn: int,
    ) -> str:
        """Send a streaming real inference request and log TTFT/TPOT.

        Returns the assistant text so the caller can append it to the
        conversation history when running the S2/S3 background traces.
        """
        self.real_sent += 1
        start = time.perf_counter()
        first_token = None
        text = ""
        usage = None
        success = False
        err = None
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
                    if first_token is None:
                        first_token = time.perf_counter()
                    text += chunk.choices[0].delta.content
                if chunk.usage:
                    usage = chunk.usage
            success = True
        except Exception as e:
            err = str(e)
        end = time.perf_counter()
        ttft = (first_token - start) if first_token else (end - start)
        total = end - start
        completion_tokens = usage.completion_tokens if usage else 0
        if completion_tokens > 1 and first_token is not None:
            tpot = (end - first_token) / (completion_tokens - 1)
        else:
            tpot = 0.0
        if success:
            self.real_success += 1
            self.real_ttft_ms.append(ttft * 1000)
            self.real_tpot_ms.append(tpot * 1000)
        await self._record(
            {
                "kind": "real",
                "tag": tag,
                "chat_id": chat_id,
                "turn": turn,
                "ttft_ms": ttft * 1000,
                "tpot_ms": tpot * 1000,
                "total_ms": total * 1000,
                "completion_tokens": completion_tokens,
                "success": success,
                "error": err,
                "ts": time.time(),
            }
        )
        return text


# ---------------------------------------------------------------------------
# Workload drivers


def _build_attack_prefix_pool(
    path: str,
    word_list: list[str],
    tokenizer,
    input_token_budget: int,
    max_prefixes: int | None = None,
) -> list[list[dict]]:
    """Build attack prefixes as proper multi-turn messages arrays.

    Each pool entry is a list of OpenAI-style ``{role, content}`` messages
    that mirrors *exactly* the structure ``prefetch_ab_runner.py`` builds
    for a real prefetch at the deepest turn of a conversation chain:

        [user_1, assistant_1, user_2, assistant_2, ..., user_k]

    Sending this through the chat-completion endpoint goes through the same
    chat template as the benign path, producing the same token sequence —
    so prefix-cache behaviour during S1 matches that of the benign trace
    instead of being a flattened, unrelated string.

    Two input formats are accepted:

    1. **Real metadata trace** (default; same file S2/S3 use, e.g.
       ``pcie_stress_heavy.jsonl``). For every multi-turn chain we
       synthesise the cumulative history through its deepest turn using
       the same word-pool rule as ``prefetch_ab_runner.py`` (input_length
       drives user-message size, output_length drives placeholder
       assistant-message size, capped at 256 tokens).
    2. **Content JSONL** (legacy ``synth_attack_prefix_8k.jsonl``): each
       entry becomes a single-turn ``[{user, content}]`` array, used only
       for the "trace-derived vs fully-synthetic" ablation.

    The runner adds a unique per-iteration suffix to the last user message
    (see ``_attack_messages``) so each attack lands a small fresh tail
    past the matched prefix, forcing actual block allocation regardless
    of cache state.
    """
    pool: list[list[dict]] = []
    if not path:
        return pool
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return pool
    first = json.loads(first_line)
    has_content = any(k in first for k in ("content", "prefix", "text"))
    has_trace = "chat_id" in first and "input_length" in first

    if has_content:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                content = rec.get("content") or rec.get("prefix") or rec.get("text")
                if content:
                    pool.append([{"role": "user", "content": content}])
    elif has_trace:
        records = _load_trace(path)
        chains = _build_chains(records)
        for chain in chains:
            messages: list[dict] = []
            for record in chain:
                user_tokens = max(
                    10, record.get("input_length", 0) - input_token_budget
                )
                user_msg = _gen_text(word_list, user_tokens, tokenizer)
                messages.append({"role": "user", "content": user_msg})
                if record is not chain[-1]:
                    # Placeholder assistant turn between user turns; size it
                    # against the recorded output_length, capped to keep the
                    # prefix manageable.
                    output_tokens = min(record.get("output_length", 64), 256)
                    placeholder = _gen_text(word_list, output_tokens, tokenizer)
                    messages.append(
                        {"role": "assistant", "content": placeholder}
                    )
            if messages:
                pool.append(messages)
    else:
        raise SystemExit(
            f"--attack-prefix-file {path}: cannot recognise schema; "
            "expected content/prefix/text or chat_id/input_length keys"
        )

    if max_prefixes is not None and len(pool) > max_prefixes:
        pool = pool[:max_prefixes]
    return pool


def _attack_messages(
    word_list: list[str],
    tokenizer,
    prefix_pool: list[list[dict]],
    idx: int,
) -> list[dict]:
    """Render a multi-turn attack request for iteration ``idx``.

    Picks a base messages array from the pool (cycling), then **appends a
    unique ~24-token suffix to the final user message** so that:

      - the leading turns hit the GPU / CPU prefix cache when revisiting
        a previously-seen chain (matching the benign-path behaviour);
      - the trailing suffix never matches anything, forcing the engine
        to allocate fresh blocks and exercise the admission-control,
        quota and TTL paths the experiment is designed to measure.
    """
    base = prefix_pool[idx % len(prefix_pool)]
    suffix_text = _gen_text(word_list, 24, tokenizer) + f" idx{idx}"
    if not base:
        return [{"role": "user", "content": suffix_text}]
    out = [dict(m) for m in base[:-1]]
    last = dict(base[-1])
    if last.get("role") == "user":
        last["content"] = f"{last['content']} {suffix_text}"
        out.append(last)
    else:
        # Defensive: chains always end with a user turn, but if not we
        # add the suffix as a fresh user turn.
        out.append(last)
        out.append({"role": "user", "content": suffix_text})
    return out


async def run_s1(args, runner: StressRunner, prefix_pool: list[list[dict]]):
    interval = 1.0 / max(args.prefetch_qps, 1e-6)
    deadline = time.time() + args.duration_sec
    pending: list[asyncio.Task] = []
    idx = 0
    next_send = time.time()
    while time.time() < deadline:
        msgs = _attack_messages(runner.word_list, runner.tokenizer, prefix_pool, idx)
        pending.append(
            asyncio.create_task(
                runner.send_prefetch(msgs, tag="s1", chat_id=-1, turn=idx)
            )
        )
        idx += 1
        next_send += interval
        wait = next_send - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        else:
            # We're falling behind the target QPS; reset rather than spin.
            next_send = time.time()
        # Periodically reap completed tasks to bound memory.
        if len(pending) > 4096:
            pending = [t for t in pending if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def run_s2(args, runner: StressRunner, chains: list[list[dict]]):
    """S2 = user mis-click burst.

    Each "user" walks a real-trace chain. At any turn that has prior history
    we issue ``--burst-size`` prefetch requests (each with a unique tail
    suffix so the engine has to allocate fresh tail blocks past the cached
    prefix), then with probability ``--abandon-prob`` we drop the rest of
    the chain *without* sending any real inference. This matches the
    reviewer's definition of misclick: "大量预取被触发但最终未提交请求".

    With ``--abandon-prob=1.0`` (the default) every chain abandons after
    its first burst, so no prefetch ever leads to a real inference. Set
    ``--abandon-prob=0`` to recover the legacy "burst then commit" mode
    used for "user hesitates but ultimately submits" experiments.

    Chains are cycled via ``itertools.cycle`` so the experiment always
    runs for the full ``--duration-sec`` regardless of trace size.
    """
    import itertools

    burst_n = args.burst_size
    burst_gap = args.burst_interval_ms / 1000.0
    abandon_prob = args.abandon_prob
    deadline = time.time() + args.duration_sec
    interval = 1.0 / max(args.qps, 1e-6)
    next_send = time.time()
    tasks: list[asyncio.Task] = []
    chain_iter = itertools.cycle(chains) if chains else iter(())
    chain_seq = 0

    async def _process_chain(chain: list[dict], seq: int):
        history: list[dict] = []
        for record in chain:
            if time.time() > deadline:
                return
            chat_id = record["chat_id"]
            turn = record["turn"]
            if history and burst_n > 0:
                msgs_for_prefetch = history.copy()
                for k in range(burst_n):
                    # Unique tail per burst so each prefetch lands a fresh
                    # tail allocation past the (typically cached) history
                    # prefix.
                    tail = f"misclick {seq}.{k} " + _gen_text(
                        runner.word_list, 16, runner.tokenizer
                    )
                    msgs = msgs_for_prefetch + [
                        {"role": "user", "content": tail}
                    ]
                    await runner.send_prefetch(
                        msgs, tag="s2_burst", chat_id=chat_id, turn=turn
                    )
                    if k < burst_n - 1:
                        await asyncio.sleep(burst_gap)
                # After the burst, decide whether to abandon. Default:
                # always abandon (matches reviewer's "未提交请求").
                if random.random() < abandon_prob:
                    return
            user_tokens = max(10, record["input_length"] - args.input_token_budget)
            user_msg = _gen_text(runner.word_list, user_tokens, runner.tokenizer)
            history.append({"role": "user", "content": user_msg})
            output_tokens = min(record.get("output_length", 64), args.max_output_tokens)
            text = await runner.send_real(
                history,
                output_tokens,
                tag="s2_real",
                chat_id=chat_id,
                turn=turn,
            )
            if text:
                history.append({"role": "assistant", "content": text})
            else:
                history.append(
                    {
                        "role": "assistant",
                        "content": _gen_text(
                            runner.word_list, output_tokens, runner.tokenizer
                        ),
                    }
                )

    while time.time() < deadline:
        try:
            chain = next(chain_iter)
        except StopIteration:
            break
        seq = chain_seq
        chain_seq += 1
        tasks.append(asyncio.create_task(_process_chain(chain, seq)))
        next_send += interval
        wait = next_send - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def run_s3(args, runner: StressRunner, chains, prefix_pool):
    """S3 = S1 attack + parallel benign trace.

    The mix-ratio controls attack QPS relative to benign QPS:
      attack_qps = args.qps * args.mix_ratio
    """
    attack_args = argparse.Namespace(**vars(args))
    attack_args.prefetch_qps = max(args.qps * args.mix_ratio, 1e-6)
    attack_args.duration_sec = args.duration_sec
    bg_args = argparse.Namespace(**vars(args))
    bg_args.burst_size = 1  # no bursting in benign workload
    await asyncio.gather(
        run_s1(attack_args, runner, prefix_pool),
        run_s2(bg_args, runner, chains),
        return_exceptions=True,
    )


# ---------------------------------------------------------------------------
# Entry point


def _resolve_tokenizer(model: str):
    """Pick a tokenizer for ``_gen_text`` length sizing.

    Priority (offline-friendly, no internet required):
      1. ``transformers.AutoTokenizer.from_pretrained(model, local_files_only=True)``
         when ``model`` is a local path. Matches what vLLM itself uses and
         is always available because the model directory is on disk.
      2. ``tiktoken.get_encoding("cl100k_base")`` if a cached BPE blob is
         already present on disk (``TIKTOKEN_CACHE_DIR`` or
         ``/tmp/data-gym-cache``). Only succeeds when populated previously.
      3. ``None`` — ``_gen_text`` falls back to a word-count heuristic
         (consistent with ``prefetch_ab_runner.py``'s tokenizer-failure
         path). Length precision degrades by ~10%, acceptable for stress
         experiments.

    Returns an object exposing ``.encode(text) -> list[int]``; for HF
    tokenizers we wrap it to suppress special tokens during length counts.
    """
    if model and os.path.isdir(model):
        try:
            from transformers import AutoTokenizer  # type: ignore

            hf_tok = AutoTokenizer.from_pretrained(
                model, local_files_only=True, trust_remote_code=True
            )

            class _HFAdapter:
                def encode(self, text: str) -> list[int]:
                    return hf_tok.encode(text, add_special_tokens=False)

            print(f"[tokenizer] using local HF tokenizer at {model}")
            return _HFAdapter()
        except Exception as e:
            print(f"[tokenizer] HF load failed ({e}); trying tiktoken")

    if tiktoken is not None:
        try:
            tk = tiktoken.get_encoding("cl100k_base")
            print("[tokenizer] using tiktoken cl100k_base (cached)")
            return tk
        except Exception as e:
            print(f"[tokenizer] tiktoken unavailable ({e}); using heuristic")

    print("[tokenizer] using len//4 heuristic (no tokenizer available)")
    return None


async def main_async(args: argparse.Namespace) -> None:
    word_list = _build_word_list(args.seed)
    tokenizer = _resolve_tokenizer(args.model)

    prefix_pool: list[list[dict]] = []
    if args.scenario in ("s1", "s3"):
        attack_src = args.attack_prefix_file or args.bg_trace_file
        if not attack_src:
            raise SystemExit(
                "--attack-prefix-file or --bg-trace-file must be provided "
                "for s1/s3 (typically the same heavy trace used for S2/S3)"
            )
        prefix_pool = _build_attack_prefix_pool(
            attack_src,
            word_list,
            tokenizer,
            input_token_budget=args.input_token_budget,
            max_prefixes=args.max_attack_prefixes,
        )
        if not prefix_pool:
            raise SystemExit(
                f"No usable entries derived from attack source {attack_src}"
            )

    chains: list[list[dict]] = []
    if args.scenario in ("s2", "s3"):
        records = _load_trace(args.bg_trace_file, args.max_chains_records)
        chains = _build_chains(records)
        if not chains:
            raise SystemExit(
                f"No multi-turn chains in bg-trace-file {args.bg_trace_file}"
            )

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)

    metrics_url = args.api_base.rstrip("/")
    if metrics_url.endswith("/v1"):
        metrics_url = metrics_url[: -len("/v1")]
    scraper = MetricsScraper(metrics_url, period_sec=args.metrics_period_sec)
    scraper.start()

    out_lock = asyncio.Lock()
    t_start = time.time()
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        runner = StressRunner(
            args.api_base,
            args.model,
            word_list,
            tokenizer,
            out_f,
            out_lock,
        )
        if args.scenario == "s1":
            await run_s1(args, runner, prefix_pool)
        elif args.scenario == "s2":
            await run_s2(args, runner, chains)
        elif args.scenario == "s3":
            await run_s3(args, runner, chains, prefix_pool)
    t_end = time.time()
    await scraper.stop()

    summary = {
        "scenario": args.scenario,
        "duration_sec": round(t_end - t_start, 3),
        "config": {
            "prefetch_qps": args.prefetch_qps if args.scenario == "s1" else None,
            "qps": args.qps if args.scenario in ("s2", "s3") else None,
            "burst_size": args.burst_size if args.scenario == "s2" else None,
            "burst_interval_ms": args.burst_interval_ms if args.scenario == "s2" else None,
            "mix_ratio": args.mix_ratio if args.scenario == "s3" else None,
        },
        "counters": {
            "prefetch_sent": runner.prefetch_sent,
            "prefetch_ack": runner.prefetch_ack,
            "prefetch_rejected_client_view": runner.prefetch_rejected,
            "real_sent": runner.real_sent,
            "real_success": runner.real_success,
        },
        "real_latency_ms": {
            "ttft_mean": (
                statistics.fmean(runner.real_ttft_ms) if runner.real_ttft_ms else None
            ),
            "ttft_p50": (
                statistics.median(runner.real_ttft_ms) if runner.real_ttft_ms else None
            ),
            "ttft_p95": (
                statistics.quantiles(runner.real_ttft_ms, n=20)[-1]
                if len(runner.real_ttft_ms) >= 20
                else None
            ),
            "tpot_mean": (
                statistics.fmean(runner.real_tpot_ms) if runner.real_tpot_ms else None
            ),
        },
        "prometheus_samples": scraper.samples,
    }
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"[done] scenario={args.scenario} "
        f"prefetch_sent={runner.prefetch_sent} "
        f"real_sent={runner.real_sent} "
        f"duration={t_end - t_start:.1f}s -> {args.output_summary}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=["s1", "s2", "s3"])
    p.add_argument("--api-base", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration-sec", type=int, default=180)
    p.add_argument("--metrics-period-sec", type=float, default=1.0)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-summary", required=True)
    # S1 / S3
    p.add_argument(
        "--prefetch-qps",
        type=float,
        default=200.0,
        help="prefetch-only requests per second (S1; S3 uses qps*mix_ratio)",
    )
    p.add_argument(
        "--attack-prefix-file",
        default="",
        help=(
            "JSONL source for S1/S3 prefixes. Accepts either a metadata "
            "trace (same format as the heavy/optimal datasets, e.g. "
            "pcie_stress_heavy.jsonl) or a content JSONL with content/"
            "prefix/text fields. If omitted, falls back to --bg-trace-file."
        ),
    )
    p.add_argument(
        "--max-attack-prefixes",
        type=int,
        default=None,
        help="Cap the prefix pool size after derivation (default: all).",
    )
    # S2 / S3
    p.add_argument("--bg-trace-file", default="")
    p.add_argument("--qps", type=float, default=1.0, help="benign trace QPS")
    p.add_argument("--burst-size", type=int, default=5)
    p.add_argument("--burst-interval-ms", type=int, default=50)
    p.add_argument(
        "--abandon-prob",
        type=float,
        default=1.0,
        help=(
            "Probability that a chain in S2 is abandoned after each burst "
            "(no follow-up real inference). 1.0 = pure misclick (matches "
            "reviewer's '大量预取被触发但最终未提交请求'); 0.0 = always "
            "commit (legacy 'hesitate then submit')."
        ),
    )
    p.add_argument("--max-output-tokens", type=int, default=128)
    p.add_argument("--input-token-budget", type=int, default=512)
    p.add_argument(
        "--max-chains-records",
        type=int,
        default=None,
        help="trim trace records before building chains (debug helper)",
    )
    # S3
    p.add_argument("--mix-ratio", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
