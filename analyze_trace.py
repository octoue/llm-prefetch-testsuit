import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

def analyze_conversation_trace(file_path: str):
    """
    分析对话trace数据
    """
    print(f"正在读取文件: {file_path}")
    
    # 存储所有记录
    records = []
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
    
    print(f"总共读取了 {len(records)} 条记录")
    
    # 构建对话树结构
    # chat_id -> record
    chat_dict = {r['chat_id']: r for r in records}
    
    # 找出每个chat_id的所有子对话
    children_dict = defaultdict(list)
    for record in records:
        parent_id = record['parent_chat_id']
        if parent_id != -1:
            children_dict[parent_id].append(record['chat_id'])
    
    # 1. 分析单轮对话占比
    print("\n" + "="*60)
    print("1. 单轮对话分析")
    print("="*60)
    
    single_turn_conversations = []
    multi_turn_conversations = []
    
    for record in records:
        chat_id = record['chat_id']
        parent_id = record['parent_chat_id']
        
        # 判断是否为单轮对话：parent=-1 且没有子对话
        if parent_id == -1 and len(children_dict[chat_id]) == 0:
            single_turn_conversations.append(chat_id)
        elif parent_id == -1:  # 是根节点，且有子对话
            multi_turn_conversations.append(chat_id)
    
    total_root_conversations = len(single_turn_conversations) + len(multi_turn_conversations)
    single_turn_ratio = len(single_turn_conversations) / total_root_conversations * 100
    
    print(f"单轮对话数量: {len(single_turn_conversations)}")
    print(f"多轮对话数量: {len(multi_turn_conversations)}")
    print(f"单轮对话占比: {single_turn_ratio:.2f}%")
    print(f"多轮对话占比: {100 - single_turn_ratio:.2f}%")
    
    # 2. 分析多轮对话的轮次间隔时间
    print("\n" + "="*60)
    print("2. 多轮对话轮次间隔时间分析")
    print("="*60)
    
    time_intervals = []
    
    for root_id in multi_turn_conversations:
        # 获取整个对话链
        conversation_chain = []
        
        # 使用BFS遍历整个对话树
        queue = [root_id]
        visited = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id in chat_dict:
                conversation_chain.append(chat_dict[current_id])
                # 添加子对话到队列
                queue.extend(children_dict[current_id])
        
        # 按照timestamp排序
        conversation_chain.sort(key=lambda x: x['timestamp'])
        
        # 计算相邻轮次的时间间隔
        for i in range(1, len(conversation_chain)):
            interval = conversation_chain[i]['timestamp'] - conversation_chain[i-1]['timestamp']
            time_intervals.append(interval)
    
    if time_intervals:
        mean_interval = np.mean(time_intervals)
        std_interval = np.std(time_intervals)
        median_interval = np.median(time_intervals)
        min_interval = np.min(time_intervals)
        max_interval = np.max(time_intervals)
        
        print(f"轮次间隔时间统计:")
        print(f"  平均值: {mean_interval:.3f} 秒")
        print(f"  标准差: {std_interval:.3f} 秒")
        print(f"  中位数: {median_interval:.3f} 秒")
        print(f"  最小值: {min_interval:.3f} 秒")
        print(f"  最大值: {max_interval:.3f} 秒")
        print(f"  总间隔数: {len(time_intervals)}")
        
        # 绘制分布直方图（百分位数）
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\n  百分位数:")
        for p in percentiles:
            print(f"    {p}%: {np.percentile(time_intervals, p):.3f} 秒")
    else:
        print("没有找到多轮对话的时间间隔数据")
    
    # 3. 分析多轮对话后续轮的平均input token长度
    print("\n" + "="*60)
    print("3. 多轮对话后续轮input token长度分析")
    print("="*60)
    
    follow_up_input_lengths = []
    
    for root_id in multi_turn_conversations:
        # 获取整个对话链
        conversation_chain = []
        queue = [root_id]
        visited = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id in chat_dict:
                conversation_chain.append(chat_dict[current_id])
                queue.extend(children_dict[current_id])
        
        # 按照turn排序
        conversation_chain.sort(key=lambda x: x['turn'])
        
        # 收集后续轮（turn > 1）的input_length
        for record in conversation_chain:
            if record['turn'] > 1:
                follow_up_input_lengths.append(record['input_length'])
    
    if follow_up_input_lengths:
        mean_input = np.mean(follow_up_input_lengths)
        std_input = np.std(follow_up_input_lengths)
        median_input = np.median(follow_up_input_lengths)
        
        print(f"后续轮input token长度统计:")
        print(f"  平均值: {mean_input:.2f} tokens")
        print(f"  标准差: {std_input:.2f} tokens")
        print(f"  中位数: {median_input:.2f} tokens")
        print(f"  最小值: {np.min(follow_up_input_lengths)} tokens")
        print(f"  最大值: {np.max(follow_up_input_lengths)} tokens")
        print(f"  总样本数: {len(follow_up_input_lengths)}")
        
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\n  百分位数:")
        for p in percentiles:
            print(f"    {p}%: {np.percentile(follow_up_input_lengths, p):.2f} tokens")
    else:
        print("没有找到后续轮的input数据")
    
    # 4. 分析多轮对话的前文context总量
    print("\n" + "="*60)
    print("4. 多轮对话前文context总量分析")
    print("="*60)
    
    context_lengths = []
    
    for root_id in multi_turn_conversations:
        # 获取整个对话链
        conversation_chain = []
        queue = [root_id]
        visited = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id in chat_dict:
                conversation_chain.append(chat_dict[current_id])
                queue.extend(children_dict[current_id])
        
        # 按照turn排序
        conversation_chain.sort(key=lambda x: x['turn'])
        
        # 计算每轮的累计context（包括之前所有轮的input和output）
        cumulative_context = 0
        for i, record in enumerate(conversation_chain):
            if i > 0:  # 从第二轮开始才有前文context
                context_lengths.append(cumulative_context)
            cumulative_context += record['input_length'] + record['output_length']
    
    if context_lengths:
        mean_context = np.mean(context_lengths)
        std_context = np.std(context_lengths)
        median_context = np.median(context_lengths)
        
        print(f"前文context总量统计:")
        print(f"  平均值: {mean_context:.2f} tokens")
        print(f"  标准差: {std_context:.2f} tokens")
        print(f"  中位数: {median_context:.2f} tokens")
        print(f"  最小值: {np.min(context_lengths)} tokens")
        print(f"  最大值: {np.max(context_lengths)} tokens")
        print(f"  总样本数: {len(context_lengths)}")
        
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\n  百分位数:")
        for p in percentiles:
            print(f"    {p}%: {np.percentile(context_lengths, p):.2f} tokens")
    else:
        print("没有找到前文context数据")
    
    # 额外统计：多轮对话的轮数分布
    print("\n" + "="*60)
    print("额外统计：多轮对话轮数分布")
    print("="*60)
    
    turn_counts = []
    for root_id in multi_turn_conversations:
        # 获取整个对话链
        conversation_chain = []
        queue = [root_id]
        visited = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id in chat_dict:
                conversation_chain.append(chat_dict[current_id])
                queue.extend(children_dict[current_id])
        
        turn_counts.append(len(conversation_chain))
    
    if turn_counts:
        print(f"多轮对话轮数统计:")
        print(f"  平均轮数: {np.mean(turn_counts):.2f}")
        print(f"  标准差: {np.std(turn_counts):.2f}")
        print(f"  中位数: {np.median(turn_counts):.2f}")
        print(f"  最小轮数: {np.min(turn_counts)}")
        print(f"  最大轮数: {np.max(turn_counts)}")

if __name__ == "__main__":
    file_path = "./qwen_traceB_blksz_16.jsonl"
    analyze_conversation_trace(file_path)