# Prefetch KV Cache 评估框架 - 项目上下文

> 本文档供后续 AI 窗口快速了解项目目标、已完成工作及当前状态。

## 目标

在 vLLM 上评估 **prefetch KV cache** 的效果：通过 OpenAI 接口 `extra_body={"prefetch": True}` 在多轮对话中利用用户 Thinking Time，提前将历史上下文发送给后端预热，从而降低真实请求的 TTFT（Time To First Token）。

评估要求：
- 基于真实 trace 做 A/B 实验（baseline vs prefetch）
- 正常配置、正常压力，不强制极端配置
- 逐条落盘指标到 JSONL
- 输出 HTML 可视化报告（Plotly/Pandas）

## 硬件与模型

| 项目 | 旧环境 | 当前环境 |
|------|--------|----------|
| GPU | NVIDIA H20-96GB | NVIDIA A100-SXM4-80GB |
| 模型 | Qwen2.5-14B-Instruct | Qwen3-8B |
| 模型路径 | - | `/lpai/models/Qwen__Qwen3-8B/25-07-26-0349` |
| HuggingFace | 可用 | **不可用**，需 `HF_HUB_OFFLINE=1` |

## 已完成工作

### 1. 核心文件

| 文件 | 作用 |
|------|------|
| `prefetch_ab_runner.py` | A/B 实验 Runner：加载 trace、按 QPS 调度、baseline/prefetch 两种模式、逐条 JSONL 落盘 |
| `generate_report.py` | 读取 baseline/prefetch JSONL，生成 HTML 报告（TTFT CDF、cached_tokens、配对对比、汇总表） |
| `run_ab_test.sh` | 编排脚本：prefetch → sleep 60 → baseline → report |
| `start_vllm.sh` | vLLM 启动脚本：本地模型、HF_HUB_OFFLINE=1、KV offload 可配、VLLM_SERVER_DEV_MODE=1 |
| `prefetch_config.sh` | 共享配置（MODEL_PATH、KV_OFFLOADING_SIZE 等） |
| `sanity_check.sh` | 快速验证 vLLM 与 reset_prefix_cache 是否可用 |

### 2. 实验可重复性与隔离

- **确定性文本**：`--seed 42` 确保两次运行（prefetch 与 baseline）生成完全相同的对话文本
- **实验间隔离**：两次实验之间调用 `POST /reset_prefix_cache?reset_external=true` 清空 GPU prefix cache 和 CPU offload，确保不互相污染
- **VLLM_SERVER_DEV_MODE=1**：`start_vllm.sh` 已启用，用于暴露 reset_prefix_cache 端点

### 3. 与旧代码对齐的修改

参考旧代码 `test.py`、`full-test.sh`、`config.env`，已完成以下对齐：

- **Prefetch 时序**：采用 lead-time 模型。prefetch 在 `scheduled_time - 0.2s` 发送，真实请求在 `scheduled_time` 发送（`--prefetch-lead-time 0.2`）
- **vLLM 参数**（按 8B/A100 等比缩放）：
  - `GPU_MEMORY_UTILIZATION=0.5`
  - `NUM_GPU_BLOCKS_OVERRIDE=115`
  - `KV_OFFLOADING_SIZE=5` GiB
  - `SWAP_SPACE=256` GiB
  - `--disable-hybrid-kv-cache-manager` 始终启用
- **A/B 顺序**：先 prefetch，再 baseline
- **冷却时间**：`sleep 60`
- **单 QPS 模式**：`QPS=0.5 ./run_ab_test.sh`，结果目录 `results/YYYYMMDD_HHMMSS_qps0.5/`
- **日志**：每次运行输出重定向到 `prefetch.log` / `baseline.log`，并从 `vllm_state.log` 提取 "Avg prompt throughput"
- **Timeout**：`TIMEOUT=120 ./run_ab_test.sh` 透传 `--timeout`
- **TPOT**：在 JSONL 中记录 `tpot_ms`

### 4. 请求发送顺序（已确认与旧代码一致）

- 用 `parent_chat_id` 构建对话链
- BFS 遍历、按 `turn` 排序
- head-N 选取多轮对话（`num_multi_turn=50`）
- 全局按 `timestamp` 排序，固定 interval `i * (1/qps)` 调度
- conversation 内串行执行，上一轮 LLM 输出作为下一轮 context

## 运行步骤

1. **启动 vLLM**（需 GPU）：
   ```bash
   cd llm-prefetch-testsuit
   ./start_vllm.sh
   # 或 KV_OFFLOADING_SIZE=0 ./start_vllm.sh  # 禁用 offload
   ```

2. **运行 A/B 实验**：
   ```bash
   ./run_ab_test.sh
   # 或 QPS=0.3 NUM_CONV=30 TIMEOUT=120 ./run_ab_test.sh
   ```

3. **查看报告**：打开 `results/<timestamp>_qps0.5/report.html`

## Trace 数据

- 路径：`qwen_traceA_blksz_16.jsonl`
- 字段：`chat_id`, `parent_chat_id`, `timestamp`, `input_length`, `output_length`, `turn`, `hash_ids`
- `parent_chat_id=-1` 为首轮；多轮通过 `parent_chat_id` 链接

## 调优建议

- **KV_OFFLOADING_SIZE**：默认 5 GiB。若 vLLM 日志出现 "cannot store blocks"，说明 CPU offload 缓冲区满，需增大（如 8 GiB）
- **NUM_GPU_BLOCKS_OVERRIDE=115**：对 8B 模型在 A100 上可能过小，导致频繁驱逐。可适当增大以模拟更真实场景，同时确保仍有足够的驱逐压力让 offloading 生效
- **实验前验证**：运行 `./sanity_check.sh` 或 `python observable_prefetch_test.py` 验证 CPU→GPU 回迁是否正常工作

## 注意事项

- 当前设备无 GPU，测试需用户手动执行
- 所有 vLLM 相关参数可通过环境变量覆盖（见 `start_vllm.sh`、`prefetch_config.sh`）
