# PCIe Profiling 使用说明

基于 run-experiment 的 lite 数据集与 vLLM 启动参数，采集 KV Offload (Evict/Restore)、Prefetch (H2D) 的 PCIe 带宽数据并生成甘特图。

## 前提

- vLLM 源码位于 `../../vllm`（或通过 `config.env` 的 `VLLM_SRC` 指定）
- vLLM 已包含 PCIe 埋点（pcie_tracer、record_function、NVTX）
- 安装 plotly：`pip install plotly`（用于生成甘特图）

## 使用流程

### 1. 启动 vLLM（PCIe Profiling 模式）

```bash
cd run-experiment
./start_vllm_pcie.sh
```

与 `start_vllm.sh` 的区别：
- 自动 source `setup_pcie_env.sh`（NCCL_P2P_DISABLE=1）
- 设置 `VLLM_PCIE_TRACE=1`
- 增加 `--gpu-profiler torch --torch-profiler-dir ./profiler_output`

### 2. 运行 lite 测试（带 Profiling）

在 vLLM 启动完成后，另开终端：

```bash
cd run-experiment
./run_lite_test_pcie.sh
# 或指定 QPS: ./run_lite_test_pcie.sh 0.4
# 禁用 TensorBoard: ./run_lite_test_pcie.sh 0.4 --no-tensorboard
```

流程：
- Phase 1：调用 `/start_profile`，运行 Prefetch 模式，结束后 `/stop_profile`
- Phase 2：运行 Baseline 模式（无 profiling）
- Phase 3：生成 report.md 和 `pcie_gantt.html`

### 3. 查看结果

- **报告**：`results/lite_qps{X}_pcie/report.md`
- **PCIe 甘特图**：`results/lite_qps{X}_pcie/pcie_gantt.html`（用浏览器打开）
- **原始事件**：`results/lite_qps{X}_pcie/pcie_events_0.json`
- **Profiler 输出**：`results/lite_qps{X}_pcie/profiler_output/`（含 trace.json）

## 配置 (config.env)

| 变量 | 说明 |
|------|------|
| `VLLM_SRC` | vLLM 源码路径，默认 `../../vllm` |
| `PCIE_PROFILER_DIR` | Profiler 输出目录，默认 `./profiler_output` |

其余参数与 `run_lite_test.sh` 相同（LITE_QPS、LITE_NUM_CONV、LITE_TRACE 等）。

## 单卡说明

当前配置为单卡（`VLLM_TENSOR_PARALLEL_SIZE=1`），无 PP 跨卡通信。采集到的 PCIe 事件主要为：
- **Evict** (D2H)：KV 从 GPU 卸载到 CPU
- **Restore** (H2D)：KV 从 CPU 恢复到 GPU
- **Prefetch** (H2D)：Prefetch 请求提前加载 KV 到 GPU

若需分析 PP 的 P2P 通信，需修改 `config.env` 中 `VLLM_TENSOR_PARALLEL_SIZE` 或增加 `--pipeline-parallel-size`，并确保多卡环境。
