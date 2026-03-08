#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析日志文件，提取TTFT、TPOT和Prefix Cache命中率等指标
"""

import os
import re
import csv
from pathlib import Path

def extract_ttft_tpot_stats(file_path):
    """从日志文件中提取TTFT和TPOT统计信息"""
    stats = {
        'avg_ttft': None,
        'median_ttft': None,
        'avg_tpot': None,
        'median_tpot': None
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # 查找TTFT统计部分
    in_ttft_section = False
    for line in lines:
        if 'TTFT (Time To First Token) 统计:' in line:
            in_ttft_section = True
            continue
        if in_ttft_section:
            if '平均值:' in line:
                match = re.search(r'平均值:\s*([\d.]+)\s*ms', line)
                if match:
                    stats['avg_ttft'] = float(match.group(1))
            elif '中位数:' in line:
                match = re.search(r'中位数:\s*([\d.]+)\s*ms', line)
                if match:
                    stats['median_ttft'] = float(match.group(1))
            elif 'TPOT' in line:  # 进入TPOT部分，结束TTFT部分
                in_ttft_section = False
                break
    
    # 查找TPOT统计部分
    in_tpot_section = False
    for line in lines:
        if 'TPOT (Time Per Output Token) 统计:' in line:
            in_tpot_section = True
            continue
        if in_tpot_section:
            if '平均值:' in line:
                match = re.search(r'平均值:\s*([\d.]+)\s*ms/token', line)
                if match:
                    stats['avg_tpot'] = float(match.group(1))
            elif '中位数:' in line:
                match = re.search(r'中位数:\s*([\d.]+)\s*ms/token', line)
                if match:
                    stats['median_tpot'] = float(match.group(1))
            elif '总响应时间统计:' in line:  # 进入下一部分，结束TPOT部分
                break
    
    return stats

def extract_prefix_cache_rates(file_path):
    """从日志文件中提取最后一行的Prefix Cache命中率"""
    rates = {
        'prefix_cache_hit_rate': None,
        'external_prefix_cache_hit_rate': None
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 从后往前查找包含"Prefix cache hit rate"的最后一行
    last_prefix_line = None
    for line in reversed(lines):
        if 'Prefix cache hit rate' in line:
            last_prefix_line = line
            break
    
    if last_prefix_line:
        # 提取Prefix cache hit rate
        match1 = re.search(r'Prefix cache hit rate:\s*([\d.]+)%', last_prefix_line)
        if match1:
            rates['prefix_cache_hit_rate'] = float(match1.group(1))
        
        # 提取External prefix cache hit rate
        match2 = re.search(r'External prefix cache hit rate:\s*([\d.]+)%', last_prefix_line)
        if match2:
            rates['external_prefix_cache_hit_rate'] = float(match2.group(1))
    
    return rates

def analyze_log_file(file_path):
    """分析单个日志文件"""
    filename = os.path.basename(file_path)
    print(f"正在分析: {filename}")
    
    # 提取统计信息
    ttft_tpot_stats = extract_ttft_tpot_stats(file_path)
    cache_rates = extract_prefix_cache_rates(file_path)
    
    # 合并结果
    result = {
        'filename': filename,
        'avg_ttft_ms': ttft_tpot_stats['avg_ttft'],
        'median_ttft_ms': ttft_tpot_stats['median_ttft'],
        'avg_tpot_ms_per_token': ttft_tpot_stats['avg_tpot'],
        'median_tpot_ms_per_token': ttft_tpot_stats['median_tpot'],
        'prefix_cache_hit_rate_percent': cache_rates['prefix_cache_hit_rate'],
        'external_prefix_cache_hit_rate_percent': cache_rates['external_prefix_cache_hit_rate']
    }
    
    return result

def main():
    # 定义要分析的文件夹
    base_dir = Path(__file__).parent
    log_dirs = [
        base_dir / 'logs'
    ]
    
    # 收集所有日志文件
    log_files = []
    for log_dir in log_dirs:
        if log_dir.exists():
            for log_file in log_dir.glob('*.log'):
                log_files.append(log_file)
        else:
            print(f"警告: 文件夹 {log_dir} 不存在")
    
    if not log_files:
        print("未找到任何日志文件")
        return
    
    print(f"找到 {len(log_files)} 个日志文件\n")
    
    # 分析所有文件
    results = []
    for log_file in sorted(log_files):
        try:
            result = analyze_log_file(log_file)
            results.append(result)
        except Exception as e:
            print(f"分析文件 {log_file} 时出错: {e}")
    
    # 输出到CSV
    output_file = base_dir / 'analysis_results.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'filename',
            'avg_ttft_ms',
            'median_ttft_ms',
            'avg_tpot_ms_per_token',
            'median_tpot_ms_per_token',
            'prefix_cache_hit_rate_percent',
            'external_prefix_cache_hit_rate_percent'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n分析完成！结果已保存到: {output_file}")
    print(f"共分析了 {len(results)} 个文件")

if __name__ == '__main__':
    main()
