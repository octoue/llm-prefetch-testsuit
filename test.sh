python workload_generator.py \
    --trace-file qwen_traceA_blksz_16.jsonl \
    --api-base http://localhost:8000/v1 \
    --num-single-turn 0 \
    --num-multi-turn 50 \
    --qps 1 \
    --model /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct \
    --output results_14b_5000.json


    # /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct
    # --model Qwen/Qwen2.5-14B-Instruct \
    
    # --preserve-timing \