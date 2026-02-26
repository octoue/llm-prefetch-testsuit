HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0,1 vllm serve --model Qwen/Qwen2.5-72B-Instruct --host 0.0.0.0 --port 8000 --max-num-seqs 256 --block-size 16 --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.95  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": 25600}}' --enable-prefix-caching --trust-remote-code --disable-hybrid-kv-cache-manager | tee vllm_state.log

# --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": <num_cpu_blocks>}}'

# /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct

# --kv-offloading-backend native --kv_offloading_size 8 