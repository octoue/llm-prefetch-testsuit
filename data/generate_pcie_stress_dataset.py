#!/usr/bin/env python3
"""
PCIe压力测试数据集生成器
生成能产生高PCIe带宽争抢但稳定运行的测试数据集
"""

import json
import random
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple


def load_conversations(trace_file: str) -> Dict[int, List[Dict]]:
    """加载并构建对话树"""
    records = []
    with open(trace_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # 构建对话树
    chat_dict = {r['chat_id']: r for r in records}
    children_dict = defaultdict(list)

    for r in records:
        parent_id = r.get('parent_chat_id', -1)
        if parent_id != -1:
            children_dict[parent_id].append(r['chat_id'])

    # 找到所有根节点
    roots = [r for r in records if r.get('parent_chat_id', -1) == -1 and children_dict[r['chat_id']]]

    # 为每个根构建完整对话链
    conversations = {}
    for root in roots:
        chain = build_conversation_chain(root['chat_id'], chat_dict, children_dict)
        if chain:
            conversations[root['chat_id']] = chain

    return conversations


def build_conversation_chain(root_id: int, chat_dict: Dict, children_dict: Dict) -> List[Dict]:
    """BFS构建对话链"""
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

    chain.sort(key=lambda x: x.get('turn', 0))
    return chain


def calculate_heaviness(conversation: List[Dict]) -> float:
    """
    计算对话的"重度",用于排序。

    重度 = 加权平均input_length + total_tokens / 100
    目标是选择能产生较大PCIe压力的对话
    """
    if not conversation:
        return 0.0

    input_lengths = [turn['input_length'] for turn in conversation]
    total_tokens = sum(turn['input_length'] + turn['output_length'] for turn in conversation)

    avg_input = sum(input_lengths) / len(input_lengths)
    total_score = min(total_tokens / 100.0, 100.0)
    turns_score = len(conversation) * 5

    return avg_input + total_score + turns_score


def filter_conversations(
    conversations: Dict[int, List[Dict]],
    min_turns: int,
    max_turns: int,
    max_single_turn_input: int,
    max_total_tokens: int
) -> List[Tuple[int, List[Dict]]]:
    """过滤出符合条件的对话"""
    filtered = []

    for conv_id, turns in conversations.items():
        num_turns = len(turns)

        if not (min_turns <= num_turns <= max_turns):
            continue

        max_input = max(turn['input_length'] for turn in turns)
        if max_input > max_single_turn_input:
            continue

        total_tokens = sum(turn['input_length'] + turn['output_length'] for turn in turns)
        if total_tokens > max_total_tokens:
            continue

        filtered.append((conv_id, turns))

    return filtered


def calculate_pcie_pressure_score(conversations: List[List[Dict]], gpu_blocks: int) -> Dict:
    """
    估算数据集产生的PCIe压力。
    """
    block_size = 16

    max_blocks_per_conv = []
    total_h2d_ops = 0

    for conv in conversations:
        max_tokens = 0
        cumulative = 0

        for turn in conv:
            cumulative += turn['input_length'] + turn['output_length']
            max_tokens = max(max_tokens, cumulative)
            total_h2d_ops += 1

        blocks_needed = (max_tokens + block_size - 1) // block_size
        max_blocks_per_conv.append(blocks_needed)

    peak_blocks = sum(max_blocks_per_conv)
    pressure_ratio = peak_blocks / gpu_blocks if gpu_blocks > 0 else float('inf')

    estimated_d2h_ops = 0
    if pressure_ratio > 1.0:
        avg_evict_size = sum(max_blocks_per_conv) / len(max_blocks_per_conv)
        excess_blocks = peak_blocks - gpu_blocks
        estimated_d2h_ops = int(excess_blocks / avg_evict_size)

    return {
        'peak_blocks_needed': peak_blocks,
        'gpu_blocks_configured': gpu_blocks,
        'pressure_ratio': pressure_ratio,
        'h2d_operations': total_h2d_ops,
        'd2h_operations': estimated_d2h_ops,
        'num_conversations': len(conversations),
        'total_turns': sum(len(conv) for conv in conversations),
    }


def generate_dataset(
    trace_file: str,
    output_path: str,
    num_conversations: int,
    turns_range: Tuple[int, int],
    input_length_max: int,
    total_tokens_max: int,
    gpu_blocks: int,
    sampling_mode: str = 'heavy',
    seed: int = 42
):
    """生成PCIe压力测试数据集"""
    print(f"[1/5] 加载原始数据集: {trace_file}")
    conversations = load_conversations(trace_file)
    print(f"      识别到 {len(conversations)} 个独立对话")

    print(f"\n[2/5] 过滤对话 (turns: {turns_range}, max_input: {input_length_max})")
    min_turns, max_turns = turns_range
    filtered = filter_conversations(
        conversations,
        min_turns=min_turns,
        max_turns=max_turns,
        max_single_turn_input=input_length_max,
        max_total_tokens=total_tokens_max
    )
    print(f"      过滤后剩余 {len(filtered)} 个对话")

    if len(filtered) < num_conversations:
        print(f"      ⚠️  警告: 只找到 {len(filtered)} 个符合条件的对话")
        num_conversations = len(filtered)

    print(f"\n[3/5] 采样 {num_conversations} 个对话 (mode: {sampling_mode})")
    random.seed(seed)

    if sampling_mode == 'heavy':
        filtered.sort(key=lambda x: calculate_heaviness(x[1]), reverse=True)
        selected = [turns for _, turns in filtered[:num_conversations]]
    elif sampling_mode == 'random':
        selected_pairs = random.sample(filtered, num_conversations)
        selected = [turns for _, turns in selected_pairs]
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

    print(f"\n[4/5] 分析PCIe压力")
    pressure = calculate_pcie_pressure_score(selected, gpu_blocks)
    print(f"      峰值blocks需求: {pressure['peak_blocks_needed']}")
    print(f"      GPU blocks配置: {pressure['gpu_blocks_configured']}")
    print(f"      压力比率: {pressure['pressure_ratio']:.2f}")
    print(f"      预估H2D操作: {pressure['h2d_operations']}")
    print(f"      预估D2H操作: {pressure['d2h_operations']}")

    if pressure['pressure_ratio'] < 0.5:
        print(f"      ⚠️  压力较低,可能无法充分测试PCIe争抢")
    elif pressure['pressure_ratio'] > 2.0:
        print(f"      ⚠️  压力过高,可能导致频繁的swap")

    print(f"\n[5/5] 写入数据集: {output_path}")
    with open(output_path, 'w') as f:
        for conversation in selected:
            for turn in conversation:
                f.write(json.dumps(turn, ensure_ascii=False) + '\n')

    total_samples = sum(len(conv) for conv in selected)
    print(f"      写入 {num_conversations} 个对话, 共 {total_samples} 个样本")

    # 写入metadata
    metadata = {
        'num_conversations': num_conversations,
        'total_samples': total_samples,
        'turns_range': turns_range,
        'input_length_max': input_length_max,
        'total_tokens_max': total_tokens_max,
        'sampling_mode': sampling_mode,
        'seed': seed,
        'pcie_pressure': pressure
    }

    metadata_path = output_path.replace('.jsonl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"      元数据: {metadata_path}")

    print(f"\n✅ 数据集生成完成!")
    print(f"\n推荐配置:")
    print(f"  TRACE={output_path}")
    print(f"  NUM_GPU_BLOCKS_OVERRIDE={pressure['gpu_blocks_configured']}")
    print(f"  QPS=0.8-1.2")
    print(f"  PREFETCH_LEAD_TIME=1.5-2.5")


def main():
    parser = argparse.ArgumentParser(
        description='生成PCIe压力测试数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  # 生成中等压力数据集
  python generate_pcie_stress_dataset.py \\
      --input data/qwen_traceA_blksz_16.jsonl \\
      --output data/pcie_stress_medium.jsonl \\
      --preset medium

  # 自定义参数
  python generate_pcie_stress_dataset.py \\
      --input data/qwen_traceA_blksz_16.jsonl \\
      --output data/pcie_stress_custom.jsonl \\
      --num-conversations 12 \\
      --turns-range 5 8 \\
      --max-input 1800 \\
      --max-total-tokens 15000 \\
      --gpu-blocks 1200
        """
    )

    parser.add_argument('--input', required=True, help='原始数据集路径')
    parser.add_argument('--output', required=True, help='输出数据集路径')

    parser.add_argument('--preset', choices=['lite', 'medium', 'heavy'],
                        help='使用预设配置')

    parser.add_argument('--num-conversations', type=int, default=12)
    parser.add_argument('--turns-range', nargs=2, type=int, default=[5, 8],
                        metavar=('MIN', 'MAX'))
    parser.add_argument('--max-input', type=int, default=1800)
    parser.add_argument('--max-total-tokens', type=int, default=15000)
    parser.add_argument('--gpu-blocks', type=int, default=1200)
    parser.add_argument('--sampling-mode', choices=['heavy', 'random'],
                        default='heavy')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # 应用预设
    if args.preset:
        presets = {
            'lite': {
                'num_conversations': 8,
                'turns_range': (3, 5),
                'max_input': 1000,
                'max_total_tokens': 8000,
                'gpu_blocks': 1500,
            },
            'medium': {
                'num_conversations': 12,
                'turns_range': (5, 8),
                'max_input': 1800,
                'max_total_tokens': 15000,
                'gpu_blocks': 1200,
            },
            'heavy': {
                'num_conversations': 40,
                'turns_range': (5, 12),
                'max_input': 2200,
                'max_total_tokens': 25000,
                'gpu_blocks': 1000,
            },
        }
        preset_config = presets[args.preset]
        for key, value in preset_config.items():
            setattr(args, key, value)
        print(f"使用预设配置: {args.preset}\n")

    generate_dataset(
        trace_file=args.input,
        output_path=args.output,
        num_conversations=args.num_conversations,
        turns_range=tuple(args.turns_range),
        input_length_max=args.max_input,
        total_tokens_max=args.max_total_tokens,
        gpu_blocks=args.gpu_blocks,
        sampling_mode=args.sampling_mode,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
