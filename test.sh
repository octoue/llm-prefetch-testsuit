# python -u workload_generator.py \
#     --trace-file qwen_traceA_blksz_16.jsonl \
#     --api-base http://localhost:8000/v1 \
#     --num-single-turn 0 \
#     --num-multi-turn 10 \
#     --qps 0.3 \
#     --timeout 120 \
#     --model /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct \
#     --output results_14b_0.3.json 

    # &> results_14b_0.3.log

python -u test.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8000/v1 \
    --num-single-turn 0 \
    --num-multi-turn 10 \
    --qps 0.3 \
    --timeout 120 \
    --enable-prefetch \
    --prefetch-lead-time 0.2 \
    --model /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct \
    --output prefetch_results_14b_0.3_0.2.json 
    
    # | tee prefetch_results_14b_0.3.log