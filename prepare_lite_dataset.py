#!/usr/bin/env python3
"""
轻量化数据集准备工具

从完整 trace 中按对话轮数分层采样，输出固定的小型 JSONL 文件。
保证每次测试使用相同数据，便于快速验证 prefetch 机制。

用法:
  python prepare_lite_dataset.py --trace-file qwen_traceA_blksz_16.jsonl --output lite_dataset.jsonl
  python prepare_lite_dataset.py --short 5 --medium 4 --long 3 --output lite_dataset.jsonl
"""

import json
import random
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple


def load_trace(trace_file: str) -> Tuple[List[Dict], Dict[int, List[int]], Dict[int, Dict]]:
    """加载 trace，返回 records 和 children_dict"""
    records = []
    chat_dict = {}
    children_dict = defaultdict(list)

    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                    chat_dict[record["chat_id"]] = record
                except json.JSONDecodeError:
                    continue

    for record in records:
        parent_id = record["parent_chat_id"]
        if parent_id != -1:
            children_dict[parent_id].append(record["chat_id"])

    return records, children_dict, chat_dict


def get_conversation_chain(root_id: int, chat_dict: Dict, children_dict: Dict) -> List[Dict]:
    """BFS 获取完整对话链，按 turn 排序"""
    chain = []
    queue = [root_id]
    visited = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        if current_id in chat_dict:
            chain.append(chat_dict[current_id])
            queue.extend(children_dict.get(current_id, []))
    chain.sort(key=lambda x: x["turn"])
    return chain


def main():
    parser = argparse.ArgumentParser(description="准备轻量化测试数据集")
    parser.add_argument("--trace-file", default="qwen_traceA_blksz_16.jsonl", help="完整 trace JSONL 路径")
    parser.add_argument("--output", default="lite_dataset.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--short", type=int, default=3, help="短对话（2-3 轮）选取数量")
    parser.add_argument("--medium", type=int, default=3, help="中等对话（4-8 轮）选取数量")
    parser.add_argument("--long", type=int, default=3, help="长对话（9+ 轮）选取数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，确保每次运行结果相同")
    args = parser.parse_args()

    records, children_dict, chat_dict = load_trace(args.trace_file)

    # 识别多轮对话（parent=-1 且有子节点）
    multi_turn_roots = []
    for record in records:
        chat_id = record["chat_id"]
        parent_id = record["parent_chat_id"]
        if parent_id == -1 and len(children_dict.get(chat_id, [])) > 0:
            multi_turn_roots.append(chat_id)

    # 按轮数分类
    short_convs = []   # 2-3 轮
    medium_convs = []  # 4-8 轮
    long_convs = []    # 9+ 轮

    for root_id in multi_turn_roots:
        chain = get_conversation_chain(root_id, chat_dict, children_dict)
        num_turns = len(chain)
        total_tokens = sum(r["input_length"] + r["output_length"] for r in chain)
        if 2 <= num_turns <= 3:
            short_convs.append((root_id, chain, total_tokens))
        elif 4 <= num_turns <= 8:
            medium_convs.append((root_id, chain, total_tokens))
        elif num_turns >= 9:
            long_convs.append((root_id, chain, total_tokens))

    random.seed(args.seed)

    def select_n(pool: List, n: int) -> List:
        if len(pool) <= n:
            return pool
        return random.sample(pool, n)

    selected_short = select_n(short_convs, args.short)
    selected_medium = select_n(medium_convs, args.medium)
    selected_long = select_n(long_convs, args.long)

    all_selected = selected_short + selected_medium + selected_long
    if not all_selected:
        print("错误: 没有找到符合条件的多轮对话")
        return 1

    # 收集所有需要输出的 record（保持原 trace 格式）
    # 按 root_id 排序，确保 prefetch_ab_runner 加载时 multi_turn_conversations 顺序固定
    output_records = []
    for root_id, chain, _ in sorted(all_selected, key=lambda x: x[0]):
        for record in chain:
            output_records.append(record)

    # 按 timestamp 和 turn 排序，与原 trace 风格一致
    output_records.sort(key=lambda r: (r.get("timestamp", 0), r["turn"]))

    with open(args.output, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_turns = sum(len(c) for _, c, _ in all_selected)
    total_tokens = sum(t for _, _, t in all_selected)

    print(f"轻量化数据集已生成: {args.output}")
    print(f"选取对话数: {len(all_selected)} (短={len(selected_short)}, 中={len(selected_medium)}, 长={len(selected_long)})")
    print(f"总请求数: {total_turns}")
    print(f"总 token 数(估算): {total_tokens}")
    print("-" * 50)
    for i, (root_id, chain, tokens) in enumerate(all_selected):
        tier = "短" if (root_id, chain, tokens) in [(r, c, t) for r, c, t in selected_short] else \
               "中" if (root_id, chain, tokens) in [(r, c, t) for r, c, t in selected_medium] else "长"
        print(f"  [{i+1}] chat_id={root_id}, 轮数={len(chain)}, tokens≈{tokens} ({tier})")
    print("-" * 50)
    print(f"预估运行时间 (QPS=0.8): 调度跨度≈{total_turns / 0.8:.0f}s, 实际取决于生成速度")
    return 0


if __name__ == "__main__":
    exit(main())
