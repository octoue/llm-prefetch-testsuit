# Workload Generator 使用说明

这是一个用于从Qwen trace数据集中采样对话并发送到vLLM服务器进行性能测试的工具。

## 功能特性

- ✅ 支持单轮和多轮对话采样
- ✅ 可配置QPS（每秒对话数）
- ✅ 可配置测试时长
- ✅ 测量TTFT（Time To First Token）
- ✅ 测量TPOT（Time Per Output Token）
- ✅ 完整的性能统计报告（平均值、标准差、百分位数）
- ✅ 支持保留原始多轮对话的时间间隔
- ✅ 异步并发请求
- ✅ 结果导出为JSON

## 安装依赖

```bash
pip install openai numpy
```

## 基本使用

### 1. 启动vLLM服务器

```bash
# 示例：启动vLLM服务器
vllm serve Qwen/Qwen2-7B-Instruct --port 8000
```

### 2. 运行workload测试

```bash
python workload_generator.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8000/v1 \
    --num-single-turn 50 \
    --num-multi-turn 50 \
    --qps 2.0 \
    --duration 60 \
    --output results.json
```

```bash
python workload_generator.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8000/v1 \
    --num-single-turn 0 \
    --num-multi-turn 50 \
    --qps 2.0 \
    --duration 60 \
    --model Qwen/Qwen2.5-14B-Instruct \
    --output results_14b.json
```

## 命令行参数

### 必需参数

- `--trace-file`: trace数据文件路径（JSONL格式）
- `--qps`: 目标QPS（每秒对话数，不是请求数）

### 服务器配置

- `--api-base`: vLLM API地址（默认: `http://localhost:8000/v1`）
- `--api-key`: API密钥（默认: `EMPTY`）
- `--model`: 模型名称（默认: `qwen`）

### Workload配置

- `--num-single-turn`: 采样的单轮对话数量（默认: 50）
- `--num-multi-turn`: 采样的多轮对话数量（默认: 50）
- `--max-turns`: 多轮对话最大轮数限制（默认: None，不限制）

### 测试配置

- `--duration`: 测试持续时间（秒）（默认: None，执行完所有workload）
- `--preserve-timing`: 保留多轮对话原始的时间间隔（默认: False）
- `--seed`: 随机种子（默认: 42）

### 输出配置

- `--output`: 输出JSON文件路径（保存详细结果）

## 使用示例

### 示例1：固定时长测试

测试60秒，QPS=5，采样100个单轮+50个多轮对话：

```bash
python workload_generator.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8000/v1 \
    --num-single-turn 100 \
    --num-multi-turn 50 \
    --qps 5.0 \
    --duration 60 \
    --output test_60s_qps5.json
```

### 示例2：执行完整workload

执行所有采样的对话，不限制时长：

```bash
python workload_generator.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8000/v1 \
    --num-single-turn 200 \
    --num-multi-turn 100 \
    --qps 2.0 \
    --output full_workload.json
```

### 示例3：限制多轮对话轮数

只测试多轮对话的前3轮：

```bash
python workload_generator.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8000/v1 \
    --num-single-turn 50 \
    --num-multi-turn 50 \
    --max-turns 3 \
    --qps 3.0 \
    --duration 120
```

### 示例4：保留原始时间间隔

保留多轮对话中原始的用户响应时间间隔：

```bash
python workload_generator.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8000/v1 \
    --num-multi-turn 100 \
    --num-single-turn 0 \
    --qps 1.0 \
    --preserve-timing \
    --output realistic_timing.json
```

### 示例5：高QPS压测

高QPS压力测试：

```bash
python workload_generator.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8000/v1 \
    --num-single-turn 500 \
    --num-multi-turn 200 \
    --qps 20.0 \
    --duration 300 \
    --output stress_test.json
```

## 输出报告

### 控制台输出

运行过程中会实时显示每个请求的状态：

```
✓ [15071_turn1] TTFT: 45.23ms, TPOT: 12.34ms, Total: 0.523s
✓ [15072_turn2] TTFT: 52.11ms, TPOT: 13.45ms, Total: 0.612s
✗ [15073_turn3] 请求失败: Connection timeout
```

测试结束后会打印完整报告：

```
================================================================================
性能测试报告
================================================================================

测试时长: 60.23 秒
实际QPS: 4.98 req/s

请求统计:
  总请求数: 300
  成功请求: 295
  失败请求: 5
  成功率: 98.33%
  单轮对话: 150
  多轮对话: 145

Token统计:
  总prompt tokens: 125430
  总completion tokens: 89234
  平均prompt tokens: 425.19
  平均completion tokens: 302.49

TTFT (Time To First Token) 统计:
  平均值: 48.25 ms
  标准差: 12.34 ms
  中位数: 45.67 ms
  P50: 45.67 ms
  P90: 65.23 ms
  P95: 72.45 ms
  P99: 89.12 ms
  最小值: 25.34 ms
  最大值: 123.45 ms

TPOT (Time Per Output Token) 统计:
  平均值: 13.45 ms/token
  标准差: 2.34 ms/token
  中位数: 12.89 ms/token
  P50: 12.89 ms/token
  P90: 16.23 ms/token
  P95: 17.45 ms/token
  P99: 20.12 ms/token

总响应时间统计:
  平均值: 0.523 秒
  标准差: 0.145 秒
  中位数: 0.489 秒
  P90: 0.712 秒
  P95: 0.823 秒
  P99: 1.023 秒
================================================================================
```

### JSON输出文件

使用 `--output` 参数保存的JSON文件包含所有统计数据：

```json
{
  "summary": {
    "total_requests": 300,
    "successful_requests": 295,
    "failed_requests": 5,
    "success_rate": 98.33,
    "single_turn_count": 150,
    "multi_turn_count": 145,
    "total_prompt_tokens": 125430,
    "total_completion_tokens": 89234,
    "avg_prompt_tokens": 425.19,
    "avg_completion_tokens": 302.49,
    "duration_seconds": 60.23,
    "actual_qps": 4.98
  },
  "ttft": {
    "mean_ms": 48.25,
    "std_ms": 12.34,
    "median_ms": 45.67,
    "p50_ms": 45.67,
    "p90_ms": 65.23,
    "p95_ms": 72.45,
    "p99_ms": 89.12
  },
  "tpot": {
    "mean_ms": 13.45,
    "std_ms": 2.34,
    "median_ms": 12.89,
    "p50_ms": 12.89,
    "p90_ms": 16.23,
    "p95_ms": 17.45,
    "p99_ms": 20.12
  }
}
```

## 性能指标说明

### TTFT (Time To First Token)
从发送请求到收到第一个token的时间，反映服务器的响应速度。

### TPOT (Time Per Output Token)
生成每个token的平均时间，反映生成速度。

### QPS (Queries Per Second)
这里的QPS指的是对话级别的QPS，不是请求级别。一个多轮对话包含多个请求。

## 注意事项

1. **QPS是对话级别的**：`--qps 2.0` 表示每秒启动2个对话，每个多轮对话可能包含多个请求。

2. **真实内容已被hash**：trace数据中的原始内容已被替换为hash_ids，脚本使用dummy文本，长度与原始数据一致。

3. **并发控制**：脚本使用异步IO实现高并发，但受限于vLLM服务器的处理能力。

4. **时间间隔**：
   - 默认模式：按QPS均匀发送对话
   - `--preserve-timing`：保留多轮对话中原始的用户响应时间间隔

5. **测试时长**：
   - 指定`--duration`：在指定时间内尽可能多地发送请求
   - 不指定：执行完所有采样的workload

## 故障排查

### 连接失败
```
请求失败: Connection refused
```
检查vLLM服务器是否正常运行，API地址是否正确。

### QPS达不到目标
实际QPS低于目标QPS，可能原因：
- 服务器处理能力不足
- 网络延迟过高
- 请求太大，处理时间过长

### 内存不足
如果trace文件太大（>1GB），加载时可能内存不足。可以：
- 使用更小的trace文件
- 减少采样数量

## 扩展使用

### 批量测试不同QPS

```bash
#!/bin/bash
for qps in 1 2 5 10 20; do
    python workload_generator.py \
        --trace-file qwen_traceA_blksz_16.jsonl \
        --api-base http://localhost:8000/v1 \
        --num-single-turn 100 \
        --num-multi-turn 50 \
        --qps $qps \
        --duration 60 \
        --output results_qps${qps}.json
done
```

### 对比不同模型

```bash
# 测试模型A
python workload_generator.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8000/v1 \
    --model qwen-7b \
    --qps 5.0 \
    --output results_qwen7b.json

# 测试模型B
python workload_generator.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8001/v1 \
    --model qwen-14b \
    --qps 5.0 \
    --output results_qwen14b.json
```
