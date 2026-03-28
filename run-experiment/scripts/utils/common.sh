#!/bin/bash
# 通用工具函数

function load_dataset_config() {
    local dataset="$1"

    # 转换为大写,构建变量前缀
    local prefix="DATASET_${dataset//-/_}"
    prefix="${prefix^^}"

    # 动态解析变量
    local trace_var="${prefix}_TRACE"
    local full_trace_var="${prefix}_FULL_TRACE"
    local max_input_var="${prefix}_MAX_INPUT"
    local num_conv_var="${prefix}_NUM_CONV"
    local gpu_blocks_var="${prefix}_GPU_BLOCKS"

    # 解引用
    TRACE="${!trace_var}"
    FULL_TRACE="${!full_trace_var}"
    MAX_INPUT="${!max_input_var}"
    NUM_CONV="${!num_conv_var}"
    DATASET_GPU_BLOCKS="${!gpu_blocks_var}"

    # 验证
    if [[ -z "$TRACE" ]]; then
        echo "❌ Error: Unknown dataset '$dataset'"
        echo ""
        echo "Available datasets:"
        env | grep '^DATASET_.*_TRACE=' | sed 's/DATASET_//;s/_TRACE=.*//' | tr '[:upper:]' '[:lower:]' | sed 's/_/-/g' | sort | sed 's/^/  - /'
        exit 1
    fi

    # 转换为绝对路径
    if [[ "$TRACE" != /* ]]; then
        TRACE="$(cd "$(dirname "$TRACE")" 2>/dev/null && pwd)/$(basename "$TRACE")" || TRACE="$TRACE"
    fi
    if [[ "$FULL_TRACE" != /* ]]; then
        FULL_TRACE="$(cd "$(dirname "$FULL_TRACE")" 2>/dev/null && pwd)/$(basename "$FULL_TRACE")" || FULL_TRACE="$FULL_TRACE"
    fi

    echo "✓ Loaded dataset config: $dataset"
    echo "  Trace: $TRACE"
    # pcie-full / pcie-trace-a-light 在 run_pcie_scheduling_ab.sh 中按 trace 统计多轮根后再打印 NUM_CONV
    if [[ "$dataset" != "pcie-full" && "$dataset" != "pcie-trace-a-light" ]]; then
        echo "  Num conversations: $NUM_CONV"
    fi
    echo "  GPU blocks (recommended): $DATASET_GPU_BLOCKS"
}

function check_vllm_running() {
    if ! curl -s "http://localhost:$API_PORT/health" &>/dev/null; then
        return 1
    fi
    return 0
}

function check_vllm_profiling() {
    # 简化检查：只要vLLM运行就认为OK，因为start_vllm_pcie.sh已经配置了profiler
    # 原检查逻辑：curl server_info | grep profiler，但API可能不返回profiler信息
    if ! curl -s "http://localhost:$API_PORT/health" &>/dev/null; then
        return 1
    fi

    # 尝试检查是否能访问profiler相关端点
    if curl -s -X POST "http://localhost:$API_PORT/start_profile" 2>&1 | grep -qE '(success|already|running|OK)'; then
        return 0
    fi

    # 如果无法确认，也返回成功（由start_vllm_pcie.sh保证profiler配置）
    return 0
}

function generate_dataset_if_needed() {
    local trace="$1"
    local full_trace="$2"
    local dataset_name="$3"

    if [[ -f "$trace" ]]; then
        return 0
    fi

    echo "⚠️  Dataset not found: $trace"
    echo "Generating from: $full_trace"

    # 从 dataset 名称提取 preset（pcie-trace-a-light 名中含 lite 子串，勿误判为 pcie_stress lite）
    local preset=""
    if [[ "$dataset_name" == "pcie-trace-a-light" ]]; then
        echo "❌ Error: 缺少 Trace A 分层轻量化文件: $trace"
        echo "   请在仓库根目录生成: python3 result-analysis/sample_trace_stratified.py --input data/qwen_traceA_blksz_16.jsonl --output data/qwen_traceA_blksz_16_light_stratified.jsonl --report result-analysis/qwen_traceA_blksz_16_light_report.md"
        return 1
    elif [[ "$dataset_name" == *"lite"* ]]; then
        preset="lite"
    elif [[ "$dataset_name" == *"medium"* ]]; then
        preset="medium"
    elif [[ "$dataset_name" == *"heavy"* ]]; then
        preset="heavy"
    fi

    if [[ -n "$preset" ]]; then
        python3 "$DATA_ROOT/generate_pcie_stress_dataset.py" \
            --input "$full_trace" \
            --output "$trace" \
            --preset "$preset" \
            --seed "$SEED"
    else
        echo "❌ Error: Cannot determine preset for dataset $dataset_name"
        return 1
    fi
}

function print_separator() {
    echo "============================================"
}

function print_phase() {
    local phase="$1"
    echo ""
    print_separator
    echo "$phase"
    print_separator
}
