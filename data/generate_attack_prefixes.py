#!/usr/bin/env python3
"""Build a JSONL of long-prefix templates used by the S1/S3 stress runner.

The output is a list of synthetic 8K-ish-token prompts. The runner appends
a tiny per-iteration suffix so each prefetch enters allocation rather than
falling back to an instant prefix-cache hit.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import tiktoken


WORD_POOL_SIZE = 10_000
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
]


def build_word_pool(seed: int) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(COMMON_WORDS) for _ in range(WORD_POOL_SIZE)]


def gen_prefix(pool: list[str], target_tokens: int, tokenizer, seed: int) -> str:
    rng = random.Random(seed)
    estimated = min(int(target_tokens * 1.4), WORD_POOL_SIZE)
    start = rng.randint(0, max(0, WORD_POOL_SIZE - estimated))
    text = " ".join(pool[start : start + estimated])
    while len(tokenizer.encode(text)) < target_tokens and estimated < WORD_POOL_SIZE:
        estimated += 64
        end = min(start + estimated, WORD_POOL_SIZE)
        text = " ".join(pool[start:end])
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--target-tokens", type=int, default=7800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tokenizer = tiktoken.get_encoding("cl100k_base")
    pool = build_word_pool(args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num):
            text = gen_prefix(pool, args.target_tokens, tokenizer, seed=args.seed + i)
            tokens = len(tokenizer.encode(text))
            f.write(
                json.dumps(
                    {
                        "id": i,
                        "approx_tokens": tokens,
                        "content": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {args.num} prefixes (target {args.target_tokens} tokens) -> {args.output}")


if __name__ == "__main__":
    main()
