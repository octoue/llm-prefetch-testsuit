
# 可用数据集快速参考

## 标准数据集（原有）

### pcie-lite
- **对话数**: 8个
- **样本数**: ~30个
- **特点**: 轻量级测试
- **GPU Blocks**: 1500
- **适用**: 快速验证

### pcie-medium
- **对话数**: 12个
- **样本数**: 44个
- **特点**: 中等压力
- **GPU Blocks**: 1200
- **适用**: 一般测试

### pcie-heavy (原版)
- **对话数**: 16个
- **样本数**: 118个
- **特点**: 高压力
- **GPU Blocks**: 1000
- **适用**: 压力测试

---

## 新增数据集（推荐）

### optimal ⭐ **推荐用于32B模型**
- **对话数**: 22个 (+37.5% vs heavy)
- **样本数**: 124个
- **轮数范围**: 5-8轮
- **输入上限**: 2000 tokens
- **GPU Blocks**: 550
- **理论压力**: 23.0x (实际 ~3.6x @ QPS=3.0)
- **特点**: 对话数多，压力适中，稳定性好
- **适用**: 32B模型的标准配置

```bash
bash run-experiment/scripts/pcie/run_profiling.sh optimal --qps 3.0
```

### many-short ⭐ **高频Prefetch**
- **对话数**: 25个（最多）
- **样本数**: 120个
- **轮数范围**: 4-7轮（较短）
- **输入上限**: 1800 tokens
- **GPU Blocks**: 500
- **理论压力**: 22.0x
- **特点**: 对话数最多，Prefetch频率最高
- **适用**: 测试Prefetch机制性能

```bash
bash run-experiment/scripts/pcie/run_profiling.sh many-short --qps 3.0
```

### balanced
- **对话数**: 16个
- **样本数**: 119个
- **轮数范围**: 6-10轮
- **输入上限**: 2300 tokens
- **GPU Blocks**: 550
- **理论压力**: 21.6x
- **特点**: 平衡的配置
- **适用**: 对比测试

```bash
bash run-experiment/scripts/pcie/run_profiling.sh balanced --qps 3.0
```

### 32b-optimized
- **对话数**: 20个
- **样本数**: 148个
- **轮数范围**: 6-10轮
- **输入上限**: 2200 tokens
- **GPU Blocks**: 500
- **理论压力**: 28.6x
- **特点**: 专门为32B模型优化
- **适用**: 32B模型进阶测试

```bash
bash run-experiment/scripts/pcie/run_profiling.sh 32b-optimized --qps 3.0
```

### extreme ⚠️ **激进配置**
- **对话数**: 18个
- **样本数**: 127个
- **轮数范围**: 6-10轮
- **输入上限**: 2500 tokens
- **GPU Blocks**: 450
- **理论压力**: 31.7x
- **特点**: 高压力，可能不稳定
- **适用**: 压力极限测试

```bash
bash run-experiment/scripts/pcie/run_profiling.sh extreme --qps 2.5
```

---

## 数据集对比表

| 数据集 | 对话数 | 样本数 | GPU Blocks | 理论压力 | 推荐QPS | 适用场景 |
|--------|--------|--------|-----------|---------|---------|----------|
| pcie-lite | 8 | ~30 | 1500 | 低 | 1.0-2.0 | 快速验证 |
| pcie-medium | 12 | 44 | 1200 | 中 | 1.0-2.0 | 一般测试 |
| pcie-heavy | 16 | 118 | 1000 | 高 | 2.0-3.0 | 压力测试 |
| **optimal** ⭐ | **22** | **124** | **550** | **23.0x** | **3.0-3.5** | **32B标准** |
| **many-short** ⭐ | **25** | **120** | **500** | **22.0x** | **3.0-3.5** | **高频Prefetch** |
| balanced | 16 | 119 | 550 | 21.6x | 3.0 | 平衡测试 |
| 32b-optimized | 20 | 148 | 500 | 28.6x | 2.5-3.0 | 32B进阶 |
| extreme ⚠️ | 18 | 127 | 450 | 31.7x | 2.0-2.5 | 极限测试 |

---

## 使用建议

### 对于32B模型

**推荐顺序**:
1. **optimal** - 作为标准baseline（对话多，稳定）
2. **many-short** - 测试高频Prefetch（对话最多）
3. **32b-optimized** - 进阶测试（样本更多）

**配置**:
- QPS: 3.0-3.5
- GPU Blocks: 让脚本自动使用推荐值
- 持续时间: 120s（默认）

### 对于8B模型

**推荐**:
- 使用 pcie-heavy 或 optimal
- QPS: 3.0-4.0
- GPU Blocks: 700-800

---

## 查看所有可用数据集

```bash
cd /lpai/volumes/ss-sai-bd-ga/wangshu/llm-prefetch-testsuit/run-experiment
bash scripts/pcie/run_profiling.sh xxx  # 故意输入错误的名字
```

会显示所有可用的数据集列表。

---

## 理论压力 vs 实际压力

**理论压力比**: 假设所有对话同时运行时的峰值blocks需求
```
理论压力 = (所有对话峰值blocks总和) / GPU blocks
```

**实际压力比**: 运行时真实的并发压力
```
实际压力 ≈ 理论压力 / 5~7  (对于32B模型)
```

**例如 optimal**:
- 理论压力: 23.0x
- 实际压力: ~3.6x (QPS=3.0时实际并发3.5个对话)

所以不用担心理论压力比过高（20+），实际运行时会自动降低。

---

## 生成自定义数据集

如果现有数据集不满足需求，可以生成自定义数据集：

```bash
cd /lpai/volumes/ss-sai-bd-ga/wangshu/llm-prefetch-testsuit/data

python3 generate_pcie_stress_dataset.py \
  --input qwen_traceA_blksz_16.jsonl \
  --output pcie_stress_custom.jsonl \
  --num-conversations 20 \
  --turns-range 5 9 \
  --max-input 2200 \
  --max-total-tokens 18000 \
  --gpu-blocks 550 \
  --sampling-mode heavy \
  --seed 42
```

然后在 `config/datasets.env` 中添加配置。

---

## 常见问题

### Q: 为什么选择optimal？
A:
- 对话数比heavy多37.5% (22 vs 16)
- GPU blocks配置更合理 (550 vs 1000)
- 专门针对32B模型调优
- 压力适中，稳定性好

### Q: many-short和optimal有什么区别？
A:
- many-short: 对话数更多(25)，但每个对话更短(4-7轮)
- optimal: 对话数适中(22)，每个对话稍长(5-8轮)
- many-short的Prefetch频率更高，但单次传输可能更小
- optimal的Prefetch更均衡

### Q: 理论压力20+会不会太高？
A: 不会。理论压力假设所有对话同时运行，但实际上：
- QPS=3.0时，实际并发只有3-4个对话
- 实际压力会降低5-7倍
- 所以理论23x → 实际3.6x（合理）

### Q: 应该用多大的GPU blocks？
A: 让脚本自动使用数据集推荐值即可：
- optimal: 550
- many-short: 500
- 32b-optimized: 500
- extreme: 450

---

**更新时间**: 2026-03-19
**作者**: Claude Code
