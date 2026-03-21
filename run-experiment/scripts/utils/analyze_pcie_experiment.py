#!/usr/bin/env python3
"""
PCIe实验结果自动分析工具

自动分析实验结果，生成详细的markdown报告，包括：
- PCIe带宽统计（H2D/D2H/P2P）
- GPU Blocks使用分布
- Prefetch行为分析
- 系统稳定性评估
- 与baseline的对比（如果提供）

用法:
  python analyze_pcie_experiment.py <results_dir> [--baseline baseline_dir] [--output report.md]

示例:
  python analyze_pcie_experiment.py results/pcie_heavy_qps3.0_dur120
  python analyze_pcie_experiment.py results/32b_550_pcie-heavy_qps3.0 --baseline results/32b_450_pcie-heavy_qps3.0
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class ExperimentAnalyzer:
    """实验结果分析器"""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.pcie_events = None
        self.vllm_log_data = None
        self.config = {}
        self.run_log_data = None

    def load_data(self):
        """加载所有实验数据"""
        # 加载PCIe事件
        pcie_file = self.results_dir / "pcie_events.json"
        if pcie_file.exists():
            with open(pcie_file) as f:
                self.pcie_events = json.load(f)

        # 加载vLLM日志
        vllm_log = self.results_dir / "vllm_state.log"
        if vllm_log.exists():
            with open(vllm_log) as f:
                self.vllm_log_data = f.read()

        # 加载run.log
        run_log = self.results_dir / "run.log"
        if run_log.exists():
            with open(run_log) as f:
                self.run_log_data = f.read()

        # 加载配置
        config_file = self.results_dir / "config_snapshot.env"
        if config_file.exists():
            with open(config_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            self.config[key.strip()] = value.strip()

    def analyze_pcie_events(self) -> Dict:
        """分析PCIe事件"""
        if not self.pcie_events:
            return {}

        # 分类统计
        h2d_events = [e for e in self.pcie_events if e.get('direction') == 'H2D']
        d2h_events = [e for e in self.pcie_events if e.get('direction') == 'D2H']
        p2p_events = [e for e in self.pcie_events if e.get('direction') == 'P2P']

        # 计算总量
        h2d_total = sum(e['size_bytes'] for e in h2d_events) / (1024**3)
        d2h_total = sum(e['size_bytes'] for e in d2h_events) / (1024**3)
        p2p_total = sum(e['size_bytes'] for e in p2p_events) / (1024**3)

        # 计算时间范围
        if self.pcie_events:
            start_time = min(e['start_us'] for e in self.pcie_events) / 1_000_000
            end_time = max((e['start_us'] + e['duration_ms']*1000) for e in self.pcie_events) / 1_000_000
            duration = end_time - start_time
        else:
            duration = 0

        # H2D事件详情
        prefetch_h2d = [e for e in h2d_events if 'Prefetch' in e.get('op_type', '')]
        restore_h2d = [e for e in h2d_events if 'Restore' in e.get('op_type', '')]

        # 事件大小分布
        h2d_sizes = sorted([e['size_mb'] for e in h2d_events]) if h2d_events else []
        d2h_sizes = sorted([e['size_mb'] for e in d2h_events]) if d2h_events else []

        return {
            'h2d_count': len(h2d_events),
            'h2d_total_gb': h2d_total,
            'd2h_count': len(d2h_events),
            'd2h_total_gb': d2h_total,
            'p2p_count': len(p2p_events),
            'p2p_total_gb': p2p_total,
            'duration': duration,
            'h2d_bandwidth': h2d_total/duration if duration > 0 else 0,
            'd2h_bandwidth': d2h_total/duration if duration > 0 else 0,
            'p2p_bandwidth': p2p_total/duration if duration > 0 else 0,
            'total_bandwidth': (h2d_total + d2h_total + p2p_total)/duration if duration > 0 else 0,
            'prefetch_count': len(prefetch_h2d),
            'restore_count': len(restore_h2d),
            'h2d_sizes': {
                'min': min(h2d_sizes) if h2d_sizes else 0,
                'p50': h2d_sizes[len(h2d_sizes)//2] if h2d_sizes else 0,
                'p90': h2d_sizes[int(len(h2d_sizes)*0.9)] if h2d_sizes else 0,
                'max': max(h2d_sizes) if h2d_sizes else 0,
                'mean': sum(h2d_sizes)/len(h2d_sizes) if h2d_sizes else 0,
            },
            'd2h_sizes': {
                'min': min(d2h_sizes) if d2h_sizes else 0,
                'p50': d2h_sizes[len(d2h_sizes)//2] if d2h_sizes else 0,
                'p90': d2h_sizes[int(len(d2h_sizes)*0.9)] if d2h_sizes else 0,
                'max': max(d2h_sizes) if d2h_sizes else 0,
                'mean': sum(d2h_sizes)/len(d2h_sizes) if d2h_sizes else 0,
            }
        }

    def analyze_gpu_blocks(self) -> Dict:
        """分析GPU blocks使用情况"""
        if not self.vllm_log_data:
            return {}

        blocks_data = []
        for line in self.vllm_log_data.split('\n'):
            match = re.search(r'free_blocks=(\d+)', line)
            if match:
                blocks_data.append(int(match.group(1)))

        if not blocks_data:
            return {}

        blocks_sorted = sorted(blocks_data)

        # 分段统计
        extreme = sum(1 for b in blocks_data if b < 20)
        tight = sum(1 for b in blocks_data if 20 <= b < 50)
        normal = sum(1 for b in blocks_data if 50 <= b < 100)
        loose = sum(1 for b in blocks_data if b >= 100)

        return {
            'sample_count': len(blocks_data),
            'min': min(blocks_data),
            'p10': blocks_sorted[len(blocks_data)//10],
            'p50': blocks_sorted[len(blocks_data)//2],
            'p90': blocks_sorted[int(len(blocks_data)*0.9)],
            'max': max(blocks_data),
            'mean': sum(blocks_data)/len(blocks_data),
            'extreme_pct': extreme/len(blocks_data)*100,
            'tight_pct': tight/len(blocks_data)*100,
            'normal_pct': normal/len(blocks_data)*100,
            'loose_pct': loose/len(blocks_data)*100,
        }

    def analyze_prefetch_behavior(self) -> Dict:
        """分析Prefetch行为"""
        if not self.run_log_data:
            return {}

        cpu_load_count = len(re.findall(r'CPU_LOAD', self.run_log_data))
        gpu_hit_count = len(re.findall(r'GPU_HIT', self.run_log_data))

        return {
            'cpu_load': cpu_load_count,
            'gpu_hit': gpu_hit_count,
            'total': cpu_load_count + gpu_hit_count,
            'cpu_load_ratio': cpu_load_count/(cpu_load_count + gpu_hit_count)*100 if (cpu_load_count + gpu_hit_count) > 0 else 0,
        }

    def analyze_stability(self) -> Dict:
        """分析系统稳定性"""
        result = {
            'total_requests': 0,
            'success_requests': 0,
            'failed_requests': 0,
            'success_rate': 0,
            'duration': 0,
            'block_failures': 0,
        }

        # 从run.log提取
        if self.run_log_data:
            match = re.search(r'\[完成\]\s+(\d+)\s+请求,\s+成功=(\d+),\s+失败=(\d+),\s+耗时\s+([\d.]+)s', self.run_log_data)
            if match:
                result['total_requests'] = int(match.group(1))
                result['success_requests'] = int(match.group(2))
                result['failed_requests'] = int(match.group(3))
                result['duration'] = float(match.group(4))
                result['success_rate'] = result['success_requests']/result['total_requests']*100 if result['total_requests'] > 0 else 0

        # 统计block failures
        if self.vllm_log_data:
            result['block_failures'] = len(re.findall(r'Block allocation failed', self.vllm_log_data, re.IGNORECASE))

        return result

    def generate_report(self, baseline_analyzer: Optional['ExperimentAnalyzer'] = None) -> str:
        """生成markdown格式的分析报告"""
        pcie = self.analyze_pcie_events()
        blocks = self.analyze_gpu_blocks()
        prefetch = self.analyze_prefetch_behavior()
        stability = self.analyze_stability()

        # 提取关键配置
        model_name = self.config.get('MODEL_PATH', 'Unknown').split('/')[-1] if 'MODEL_PATH' in self.config else 'Unknown'
        gpu_blocks = self.config.get('NUM_GPU_BLOCKS_OVERRIDE', 'N/A')

        lines = []
        lines.append("# PCIe实验结果分析报告")
        lines.append("")
        lines.append(f"**实验目录**: `{self.results_dir.name}`")
        lines.append(f"**生成时间**: {self._get_timestamp()}")
        lines.append("")

        # 配置信息
        lines.append("## 1. 实验配置")
        lines.append("")
        lines.append("| 配置项 | 值 |")
        lines.append("|--------|-----|")
        lines.append(f"| 模型 | {model_name} |")
        lines.append(f"| GPU Blocks | {gpu_blocks} |")
        lines.append(f"| 持续时间 | {stability['duration']:.1f}s |")
        lines.append(f"| 总请求数 | {stability['total_requests']} |")
        lines.append(f"| 成功率 | {stability['success_rate']:.1f}% |")
        lines.append("")

        # PCIe带宽统计
        lines.append("## 2. PCIe带宽统计")
        lines.append("")

        if baseline_analyzer:
            baseline_pcie = baseline_analyzer.analyze_pcie_events()
            lines.append("| 指标 | 当前值 | Baseline | 变化 |")
            lines.append("|------|--------|----------|------|")
            lines.append(f"| H2D事件数 | {pcie['h2d_count']} | {baseline_pcie['h2d_count']} | {self._format_change(pcie['h2d_count'], baseline_pcie['h2d_count'])} |")
            lines.append(f"| H2D总量 (GB) | {pcie['h2d_total_gb']:.2f} | {baseline_pcie['h2d_total_gb']:.2f} | {self._format_change(pcie['h2d_total_gb'], baseline_pcie['h2d_total_gb'])} |")
            lines.append(f"| H2D带宽 (GB/s) | {pcie['h2d_bandwidth']:.3f} | {baseline_pcie['h2d_bandwidth']:.3f} | {self._format_change(pcie['h2d_bandwidth'], baseline_pcie['h2d_bandwidth'])} |")
            lines.append(f"| D2H带宽 (GB/s) | {pcie['d2h_bandwidth']:.3f} | {baseline_pcie['d2h_bandwidth']:.3f} | {self._format_change(pcie['d2h_bandwidth'], baseline_pcie['d2h_bandwidth'])} |")
            lines.append(f"| P2P带宽 (GB/s) | {pcie['p2p_bandwidth']:.3f} | {baseline_pcie['p2p_bandwidth']:.3f} | {self._format_change(pcie['p2p_bandwidth'], baseline_pcie['p2p_bandwidth'])} |")
            lines.append(f"| 总带宽 (GB/s) | {pcie['total_bandwidth']:.3f} | {baseline_pcie['total_bandwidth']:.3f} | {self._format_change(pcie['total_bandwidth'], baseline_pcie['total_bandwidth'])} |")
        else:
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| H2D事件数 | {pcie['h2d_count']} 次 |")
            lines.append(f"| H2D总量 | {pcie['h2d_total_gb']:.2f} GB |")
            lines.append(f"| H2D带宽 | {pcie['h2d_bandwidth']:.3f} GB/s |")
            lines.append(f"| D2H事件数 | {pcie['d2h_count']} 次 |")
            lines.append(f"| D2H总量 | {pcie['d2h_total_gb']:.2f} GB |")
            lines.append(f"| D2H带宽 | {pcie['d2h_bandwidth']:.3f} GB/s |")
            lines.append(f"| P2P事件数 | {pcie['p2p_count']} 次 |")
            lines.append(f"| P2P带宽 | {pcie['p2p_bandwidth']:.3f} GB/s |")
            lines.append(f"| 总带宽 | {pcie['total_bandwidth']:.3f} GB/s |")
            lines.append(f"| PCIe利用率 | {pcie['total_bandwidth']/32*100:.2f}% (假设PCIe 4.0 x16) |")

        lines.append("")

        # H2D事件详情
        lines.append("### 2.1 H2D事件详情")
        lines.append("")
        lines.append("| 类型 | 数量 |")
        lines.append("|------|------|")
        lines.append(f"| Prefetch操作 | {pcie['prefetch_count']} |")
        lines.append(f"| Restore操作 | {pcie['restore_count']} |")
        lines.append("")
        lines.append("**H2D事件大小分布 (MB)**:")
        lines.append("")
        lines.append(f"- Min: {pcie['h2d_sizes']['min']:.1f} MB")
        lines.append(f"- P50: {pcie['h2d_sizes']['p50']:.1f} MB")
        lines.append(f"- P90: {pcie['h2d_sizes']['p90']:.1f} MB")
        lines.append(f"- Max: {pcie['h2d_sizes']['max']:.1f} MB")
        lines.append(f"- Mean: {pcie['h2d_sizes']['mean']:.1f} MB")
        lines.append("")

        # GPU Blocks使用
        if blocks:
            lines.append("## 3. GPU Blocks使用情况")
            lines.append("")
            lines.append("### 3.1 Free Blocks分布")
            lines.append("")
            lines.append("| 统计量 | 值 |")
            lines.append("|--------|-----|")
            lines.append(f"| 采样点数 | {blocks['sample_count']} |")
            lines.append(f"| Min | {blocks['min']} |")
            lines.append(f"| P10 | {blocks['p10']} |")
            lines.append(f"| P50 | {blocks['p50']} |")
            lines.append(f"| P90 | {blocks['p90']} |")
            lines.append(f"| Max | {blocks['max']} |")
            lines.append(f"| Mean | {blocks['mean']:.1f} |")
            lines.append("")
            lines.append("### 3.2 压力分布")
            lines.append("")
            lines.append("| 压力级别 | 百分比 | 评价 |")
            lines.append("|----------|--------|------|")
            lines.append(f"| 极度紧张 (<20) | {blocks['extreme_pct']:.1f}% | {self._evaluate_pressure('extreme', blocks['extreme_pct'])} |")
            lines.append(f"| 紧张 (20-50) | {blocks['tight_pct']:.1f}% | {self._evaluate_pressure('tight', blocks['tight_pct'])} |")
            lines.append(f"| 正常 (50-100) | {blocks['normal_pct']:.1f}% | {self._evaluate_pressure('normal', blocks['normal_pct'])} |")
            lines.append(f"| 宽松 (≥100) | {blocks['loose_pct']:.1f}% | {self._evaluate_pressure('loose', blocks['loose_pct'])} |")
            lines.append("")

        # Prefetch行为
        if prefetch:
            lines.append("## 4. Prefetch行为分析")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| CPU_LOAD次数 | {prefetch['cpu_load']} |")
            lines.append(f"| GPU_HIT次数 | {prefetch['gpu_hit']} |")
            lines.append(f"| 总Prefetch尝试 | {prefetch['total']} |")
            lines.append(f"| CPU_LOAD比例 | {prefetch['cpu_load_ratio']:.1f}% |")

            if pcie:
                h2d_per_prefetch = pcie['h2d_count'] / prefetch['cpu_load'] if prefetch['cpu_load'] > 0 else 0
                lines.append(f"| H2D事件/CPU_LOAD | {h2d_per_prefetch:.2f} |")
            lines.append("")

        # 系统稳定性
        lines.append("## 5. 系统稳定性")
        lines.append("")
        lines.append("| 指标 | 值 | 评价 |")
        lines.append("|------|-----|------|")
        lines.append(f"| 请求成功率 | {stability['success_rate']:.1f}% | {self._evaluate_success_rate(stability['success_rate'])} |")
        lines.append(f"| 失败请求数 | {stability['failed_requests']} | {self._evaluate_failures(stability['failed_requests'])} |")
        lines.append(f"| Block allocation失败 | {stability['block_failures']} | {'✓ 都已恢复' if stability['failed_requests'] == 0 else '⚠️ 有请求失败'} |")
        lines.append(f"| 实际吞吐 | {stability['success_requests']/stability['duration']:.2f} req/s | - |")
        lines.append("")

        # 综合评估
        lines.append("## 6. 综合评估")
        lines.append("")
        lines.append(self._generate_summary(pcie, blocks, prefetch, stability))
        lines.append("")

        # 建议
        lines.append("## 7. 优化建议")
        lines.append("")
        lines.append(self._generate_recommendations(pcie, blocks, stability))
        lines.append("")

        return '\n'.join(lines)

    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _format_change(self, current, baseline):
        """格式化变化百分比"""
        if baseline == 0:
            return "N/A"
        change = (current - baseline) / baseline * 100
        sign = "+" if change >= 0 else ""
        return f"{sign}{change:.1f}%"

    def _evaluate_pressure(self, level, pct):
        """评估压力分布"""
        if level == 'extreme':
            if pct > 30:
                return "⚠️ 偏高"
            elif pct > 15:
                return "✓ 良好"
            else:
                return "- 偏低"
        elif level == 'loose':
            if pct > 30:
                return "⚠️ 过于宽松"
            elif pct < 15:
                return "✓ 良好"
            else:
                return "- 适中"
        return "-"

    def _evaluate_success_rate(self, rate):
        """评估成功率"""
        if rate >= 99.5:
            return "✓✓ 优秀"
        elif rate >= 95:
            return "✓ 良好"
        elif rate >= 90:
            return "⚠️ 需改进"
        else:
            return "❌ 不稳定"

    def _evaluate_failures(self, count):
        """评估失败数"""
        if count == 0:
            return "✓✓ 完美"
        elif count <= 5:
            return "✓ 可接受"
        else:
            return "⚠️ 偏多"

    def _generate_summary(self, pcie, blocks, prefetch, stability):
        """生成综合评估摘要"""
        lines = []

        # 评估PCIe带宽
        if pcie['h2d_bandwidth'] > 0.3:
            lines.append("✓ **PCIe带宽表现良好**: H2D带宽 {:.3f} GB/s，已产生明显的PCIe争抢".format(pcie['h2d_bandwidth']))
        elif pcie['h2d_bandwidth'] > 0.15:
            lines.append("- **PCIe带宽适中**: H2D带宽 {:.3f} GB/s，有一定的PCIe活动".format(pcie['h2d_bandwidth']))
        else:
            lines.append("⚠️ **PCIe带宽较低**: H2D带宽 {:.3f} GB/s，可能需要增加压力".format(pcie['h2d_bandwidth']))

        # 评估GPU压力
        if blocks:
            if blocks['loose_pct'] < 15 and blocks['extreme_pct'] > 15:
                lines.append("✓ **GPU内存压力合理**: {}%紧张，{}%宽松，offloading机制被充分激活".format(
                    int(blocks['extreme_pct'] + blocks['tight_pct']), int(blocks['loose_pct'])))
            elif blocks['loose_pct'] > 30:
                lines.append("⚠️ **GPU内存压力偏低**: {}%时间宽松，建议降低GPU blocks".format(int(blocks['loose_pct'])))
            else:
                lines.append("- **GPU内存压力适中**: 压力分布较为均衡")

        # 评估稳定性
        if stability['success_rate'] >= 99:
            lines.append("✓✓ **系统稳定性优秀**: 成功率 {:.1f}%，无失败请求".format(stability['success_rate']))
        elif stability['success_rate'] >= 95:
            lines.append("✓ **系统稳定性良好**: 成功率 {:.1f}%".format(stability['success_rate']))
        else:
            lines.append("⚠️ **系统稳定性需改进**: 成功率 {:.1f}%，有 {} 个请求失败".format(
                stability['success_rate'], stability['failed_requests']))

        return '\n'.join(lines)

    def _generate_recommendations(self, pcie, blocks, stability):
        """生成优化建议"""
        lines = []

        # 基于PCIe带宽的建议
        if pcie['h2d_bandwidth'] < 0.15:
            lines.append("### 提升PCIe带宽")
            lines.append("- 考虑降低GPU blocks配置（如果稳定性允许）")
            lines.append("- 增加QPS以提高并发度")
            lines.append("- 使用对话数更多的数据集")

        # 基于GPU压力的建议
        if blocks and blocks['loose_pct'] > 25:
            lines.append("### 优化GPU内存使用")
            lines.append("- 当前 {:.1f}% 时间处于宽松状态，可降低GPU blocks".format(blocks['loose_pct']))
            lines.append("- 建议将GPU blocks降低10-15%")

        # 基于稳定性的建议
        if stability['success_rate'] < 98:
            lines.append("### 提升系统稳定性")
            lines.append("- 当前有 {} 个请求失败，建议增加GPU blocks".format(stability['failed_requests']))
            lines.append("- 或降低QPS以减少并发压力")

        # 通用建议
        if not lines:
            lines.append("### 当前配置表现良好")
            lines.append("- 系统稳定且PCIe带宽合理")
            lines.append("- 可以开始测试链路管理优化算法")
            lines.append("- 或尝试更激进的数据集配置")

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='自动分析PCIe实验结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析单个实验
  python analyze_pcie_experiment.py results/pcie_heavy_qps3.0_dur120

  # 对比两个实验
  python analyze_pcie_experiment.py results/32b_550_pcie-heavy_qps3.0 \\
      --baseline results/32b_450_pcie-heavy_qps3.0

  # 指定输出文件
  python analyze_pcie_experiment.py results/32b_550_pcie-heavy_qps3.0 \\
      --output analysis_report.md
        """
    )

    parser.add_argument('results_dir', type=Path, help='实验结果目录')
    parser.add_argument('--baseline', type=Path, help='Baseline实验目录（用于对比）')
    parser.add_argument('--output', type=Path, help='输出报告文件（默认: <results_dir>/analysis_report.md）')

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"❌ 错误: 目录不存在: {args.results_dir}")
        sys.exit(1)

    # 分析当前实验
    analyzer = ExperimentAnalyzer(args.results_dir)
    analyzer.load_data()

    # 分析baseline（如果提供）
    baseline_analyzer = None
    if args.baseline:
        if not args.baseline.exists():
            print(f"⚠️  警告: Baseline目录不存在: {args.baseline}")
        else:
            baseline_analyzer = ExperimentAnalyzer(args.baseline)
            baseline_analyzer.load_data()

    # 生成报告
    report = analyzer.generate_report(baseline_analyzer)

    # 输出
    if args.output:
        output_file = args.output
    else:
        output_file = args.results_dir / "analysis_report.md"

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✅ 分析报告已生成: {output_file}")
    print("")
    print("=" * 70)
    print("关键指标摘要:")
    print("=" * 70)

    pcie = analyzer.analyze_pcie_events()
    stability = analyzer.analyze_stability()

    if pcie:
        print(f"H2D带宽: {pcie['h2d_bandwidth']:.3f} GB/s ({pcie['h2d_count']} 次事件)")
        print(f"D2H带宽: {pcie['d2h_bandwidth']:.3f} GB/s ({pcie['d2h_count']} 次事件)")
        print(f"总带宽: {pcie['total_bandwidth']:.3f} GB/s")

    if stability:
        print(f"成功率: {stability['success_rate']:.1f}% ({stability['success_requests']}/{stability['total_requests']})")
        print(f"持续时间: {stability['duration']:.1f}s")

    print("=" * 70)


if __name__ == '__main__':
    main()
