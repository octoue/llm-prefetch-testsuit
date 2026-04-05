# EPLB + PCIe Scheduling Feasibility Test

## Background

vLLM supports Expert Parallelism (EP) and Expert Parallel Load Balancing (EPLB) for MoE models.
Both EP all-to-all communication and EPLB weight migration use NCCL, which works over PCIe.

**Goal**: Validate that EP + EPLB can run on single-machine multi-GPU over PCIe (NVLink disabled),
then explore whether PCIe scheduling can coordinate prefetch/KV transfers with EP traffic.

## PCIe Feasibility Summary

| Component | PCIe Compatible? | Mechanism |
|-----------|:---:|-----------|
| EP all-to-all (allgather_reducescatter) | YES | NCCL allgather + reduce_scatter |
| EP all-to-all (naive) | YES | NCCL broadcast |
| EPLB weight migration | YES | torch.distributed P2P (isend/irecv via NCCL) |
| EPLB async mode | YES | Background thread, same NCCL P2P |

## Model Choice: DeepSeek-V2-Lite-Chat

- **Why**: 16B total params, 2.4B active params, 64 routed experts per layer
- **Fits 2x A100-80GB** with EP=2 (each GPU holds 32 experts)
- vLLM officially tests this model for EP (`tests/distributed/test_expert_parallel.py`)
- Alternative: Mixtral-8x7B (needs EP=4 or TP=4)

## Scripts

| Script | Purpose |
|--------|---------|
| `start_vllm_ep.sh` | Start vLLM with EP + EPLB + PCIe-only |
| `test_ep_basic.sh` | Quick smoke test: start server, send requests, verify EP works |
| `test_ep_eplb.sh` | EPLB test: sustained load to trigger expert rearrangement |
| `bench_ep_pcie.py` | Benchmark: measure all-to-all latency and EPLB migration overhead |

## Usage

```bash
# 1. Smoke test (2x GPU)
./test_ep_basic.sh

# 2. EPLB trigger test (sustained load)
./test_ep_eplb.sh

# 3. Benchmark EP + PCIe
python3 bench_ep_pcie.py --api-base http://localhost:8000/v1 --qps 1.0 --num-requests 100
```
