# PCIe实验自动分析工具使用说明

## 功能概述

`analyze_pcie_experiment.py` 是一个自动化的PCIe实验结果分析工具，能够：

- 📊 自动分析PCIe带宽（H2D/D2H/P2P）
- 💾 分析GPU Blocks使用分布
- 🔄 分析Prefetch行为
- ✅ 评估系统稳定性
- 📈 对比不同配置的差异
- 📝 生成详细的Markdown分析报告

## 自动化流程

### 运行实验后自动分析

当你使用 `run_profiling.sh` 运行实验时，脚本会**自动**调用分析工具：

```bash
# 运行实验（会自动生成分析报告）
bash run-experiment/scripts/pcie/run_profiling.sh pcie-heavy --qps 3.0
```

实验完成后，你会在结果目录中看到：
```
results/Qwen3-32B_blk550_pcie-heavy_qps3.0_dur120/
├── pcie_events.json          # PCIe原始事件数据
├── pcie_gantt.html           # 可视化甘特图
├── vllm_state.log            # vLLM运行日志
├── run.log                   # 实验运行日志
├── config_snapshot.env       # 配置快照
└── analysis_report.md        # ✨ 自动生成的分析报告
```

## 手动分析

### 1. 分析单个实验

```bash
python3 run-experiment/scripts/utils/analyze_pcie_experiment.py \
    results/Qwen3-32B_blk550_pcie-heavy_qps3.0_dur120
```

输出：
- `results/Qwen3-32B_blk550_pcie-heavy_qps3.0_dur120/analysis_report.md`

### 2. 对比两个实验

```bash
python3 run-experiment/scripts/utils/analyze_pcie_experiment.py \
    results/Qwen3-32B_blk550_pcie-heavy_qps3.0_dur120 \
    --baseline results/Qwen3-32B_blk450_pcie-heavy_qps3.0_dur120
```

报告中会包含详细的对比数据。

### 3. 指定输出文件

```bash
python3 run-experiment/scripts/utils/analyze_pcie_experiment.py \
    results/Qwen3-32B_blk550_pcie-heavy_qps3.0_dur120 \
    --output my_custom_report.md
```

## 分析报告内容

生成的 `analysis_report.md` 包含以下章节：

### 1. 实验配置
- 模型信息
- GPU Blocks配置
- 持续时间
- 请求统计

### 2. PCIe带宽统计
- H2D/D2H/P2P事件数量和总量
- 各类型带宽（GB/s）
- PCIe利用率
- H2D事件大小分布

### 3. GPU Blocks使用情况
- Free blocks分布（Min/P10/P50/P90/Max/Mean）
- 压力分布（极度紧张/紧张/正常/宽松的比例）

### 4. Prefetch行为分析
- CPU_LOAD vs GPU_HIT统计
- H2D事件与Prefetch的对应关系

### 5. 系统稳定性
- 请求成功率
- 失败请求统计
- Block allocation失败次数
- 实际吞吐量

### 6. 综合评估
- 自动评价各项指标
- 识别潜在问题

### 7. 优化建议
- 基于分析结果的具体建议
- 下一步实验方向

## 实验目录命名规则

新的目录命名格式：
```
<模型名称>_blk<GPU_BLOCKS>_<数据集>_qps<QPS>_dur<持续时间>
```

示例：
```
Qwen3-32B_blk550_pcie-heavy_qps3.0_dur120
Qwen3-8B_blk700_pcie-optimal_qps4.0_dur180
```

优点：
- 一目了然地看到实验配置
- 方便对比不同配置
- 避免混淆

## 快速查看分析报告

```bash
# 使用cat直接查看
cat results/Qwen3-32B_blk550_pcie-heavy_qps3.0_dur120/analysis_report.md

# 或使用markdown阅读器
glow results/Qwen3-32B_blk550_pcie-heavy_qps3.0_dur120/analysis_report.md

# 或在VSCode等编辑器中打开
code results/Qwen3-32B_blk550_pcie-heavy_qps3.0_dur120/analysis_report.md
```

## 让Claude分析报告

你可以直接让Claude读取分析报告：

```
请阅读 results/Qwen3-32B_blk550_pcie-heavy_qps3.0_dur120/analysis_report.md
```

Claude会基于报告内容给出详细的解读和建议。

## 批量分析多个实验

```bash
# 分析最近的5个实验
for dir in $(ls -td results/Qwen3-32B_blk*/ | head -5); do
    echo "Analyzing $dir..."
    python3 run-experiment/scripts/utils/analyze_pcie_experiment.py "$dir"
done
```

## 故障排查

### 问题1: 脚本找不到

```bash
# 确保在正确的目录
cd /lpai/volumes/ss-sai-bd-ga/wangshu/llm-prefetch-testsuit

# 检查脚本是否存在
ls -l run-experiment/scripts/utils/analyze_pcie_experiment.py
```

### 问题2: 没有执行权限

```bash
chmod +x run-experiment/scripts/utils/analyze_pcie_experiment.py
```

### 问题3: 缺少数据文件

确保实验结果目录包含：
- `pcie_events.json` (必需)
- `vllm_state.log` (推荐)
- `run.log` (推荐)
- `config_snapshot.env` (推荐)

## 高级用法

### 自定义分析逻辑

你可以修改 `analyze_pcie_experiment.py` 来添加自定义的分析逻辑：

```python
def analyze_custom_metric(self) -> Dict:
    """添加你的自定义分析"""
    # 分析代码...
    return result
```

### 导出为其他格式

分析脚本目前输出Markdown格式，你可以使用pandoc转换：

```bash
# 转为PDF
pandoc results/xxx/analysis_report.md -o report.pdf

# 转为HTML
pandoc results/xxx/analysis_report.md -o report.html
```

## 总结

使用这个自动分析工具，你可以：

✅ **节省时间**: 不再需要手动编写分析脚本
✅ **标准化**: 所有实验使用相同的分析标准
✅ **可对比**: 轻松对比不同配置的效果
✅ **快速迭代**: 实验-分析-优化循环更快
✅ **易于分享**: 报告格式统一，方便沟通

---

**作者**: Claude Code
**更新时间**: 2026-03-19
