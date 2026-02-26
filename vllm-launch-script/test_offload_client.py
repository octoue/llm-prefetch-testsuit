#!/usr/bin/env python3
"""
vLLM CPU Offloading 压力测试客户端

该脚本通过发送大量并发请求来触发 KV cache 的 CPU offloading，
并监控换入(load)和换出(store)事件。
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List

import aiohttp


@dataclass
class RequestResult:
    """单个请求的结果"""
    request_id: int
    prompt_tokens: int
    generated_tokens: int
    latency: float
    success: bool
    error: str = ""


class OffloadTestClient:
    """CPU Offloading 测试客户端"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 300,
    ):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.results: List[RequestResult] = []

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        request_id: int,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> RequestResult:
        """发送单个请求"""
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        start_time = time.time()
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    latency = time.time() - start_time
                    
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    
                    print(f"✓ Request {request_id}: "
                          f"{prompt_tokens} prompt + {completion_tokens} generated tokens, "
                          f"latency: {latency:.2f}s")
                    
                    return RequestResult(
                        request_id=request_id,
                        prompt_tokens=prompt_tokens,
                        generated_tokens=completion_tokens,
                        latency=latency,
                        success=True,
                    )
                else:
                    error_text = await response.text()
                    print(f"✗ Request {request_id} failed: HTTP {response.status} - {error_text}")
                    return RequestResult(
                        request_id=request_id,
                        prompt_tokens=0,
                        generated_tokens=0,
                        latency=time.time() - start_time,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}",
                    )
        except Exception as e:
            print(f"✗ Request {request_id} error: {e}")
            return RequestResult(
                request_id=request_id,
                prompt_tokens=0,
                generated_tokens=0,
                latency=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def generate_prompts(self, pattern: str, num_prompts: int, prompt_length: int = 200) -> List[str]:
        """生成测试 prompts
        
        Args:
            pattern: 'unique' 或 'shared' 或 'mixed'
                - unique: 每个请求不同的 prompt（更容易触发 offload）
                - shared: 所有请求共享前缀（测试 prefix caching）
                - mixed: 混合模式
            num_prompts: 生成的 prompt 数量
            prompt_length: 每个 prompt 的大约长度（tokens）
        """
        prompts = []
        
        if pattern == "unique":
            # 生成独特的 prompts，更容易填满 KV cache
            base_text = "Write a detailed story about "
            topics = [
                "a space explorer", "a time traveler", "a deep sea diver",
                "a mountain climber", "a detective", "a scientist",
                "an artist", "a musician", "a chef", "an architect",
                "a pilot", "a teacher", "a doctor", "a programmer",
                "a writer", "a photographer", "an athlete", "a gardener",
            ]
            for i in range(num_prompts):
                topic = topics[i % len(topics)]
                prompt = f"{base_text}{topic} #{i}. " + "Please provide rich details. " * (prompt_length // 10)
                prompts.append(prompt)
                
        elif pattern == "shared":
            # 共享前缀，测试 prefix caching
            shared_prefix = "You are a helpful assistant. Please answer the following question in detail. " * 5
            questions = [
                "What is machine learning?",
                "Explain quantum computing.",
                "How does photosynthesis work?",
                "What is the theory of relativity?",
                "Describe the water cycle.",
                "What is artificial intelligence?",
                "Explain blockchain technology.",
                "How do neural networks work?",
            ]
            for i in range(num_prompts):
                question = questions[i % len(questions)]
                prompt = shared_prefix + question + f" (Question #{i})"
                prompts.append(prompt)
                
        elif pattern == "mixed":
            # 混合模式：部分共享前缀，部分独特
            for i in range(num_prompts):
                if i % 3 == 0:
                    # 共享前缀
                    prompt = "System prompt: " * 10 + f"User query #{i}"
                else:
                    # 独特 prompt
                    prompt = f"Unique request #{i}: " + "x" * prompt_length
                prompts.append(prompt)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
            
        return prompts

    async def run_batch_test(
        self,
        num_requests: int = 50,
        concurrency: int = 10,
        prompt_pattern: str = "unique",
        max_tokens: int = 100,
        delay_between_batches: float = 0.5,
    ):
        """运行批量测试
        
        Args:
            num_requests: 总请求数
            concurrency: 并发数
            prompt_pattern: prompt 生成模式
            max_tokens: 每个请求生成的最大 token 数
            delay_between_batches: 批次之间的延迟（秒）
        """
        print(f"\n{'='*60}")
        print(f"Starting CPU Offloading Stress Test")
        print(f"{'='*60}")
        print(f"Total Requests: {num_requests}")
        print(f"Concurrency: {concurrency}")
        print(f"Prompt Pattern: {prompt_pattern}")
        print(f"Max Tokens per Request: {max_tokens}")
        print(f"{'='*60}\n")

        # 生成 prompts
        prompts = self.generate_prompts(prompt_pattern, num_requests)
        
        # 检查服务是否可用
        print("Checking if vLLM server is ready...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status != 200:
                        print(f"❌ Server health check failed: HTTP {response.status}")
                        return
            print("✓ Server is ready\n")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print(f"   Please make sure vLLM is running at {self.base_url}")
            return

        # 分批发送请求
        start_time = time.time()
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            for batch_start in range(0, num_requests, concurrency):
                batch_end = min(batch_start + concurrency, num_requests)
                batch_size = batch_end - batch_start
                
                print(f"\n--- Batch {batch_start//concurrency + 1} "
                      f"(Requests {batch_start+1}-{batch_end}) ---")
                
                # 创建并发任务
                tasks = []
                for i in range(batch_start, batch_end):
                    task = self.send_request(
                        session=session,
                        request_id=i + 1,
                        prompt=prompts[i],
                        max_tokens=max_tokens,
                    )
                    tasks.append(task)
                
                # 等待该批次完成
                batch_results = await asyncio.gather(*tasks)
                self.results.extend(batch_results)
                
                # 批次间延迟
                if batch_end < num_requests:
                    await asyncio.sleep(delay_between_batches)

        total_time = time.time() - start_time
        
        # 打印统计结果
        self.print_statistics(total_time)

    def print_statistics(self, total_time: float):
        """打印测试统计信息"""
        print(f"\n{'='*60}")
        print(f"Test Results Summary")
        print(f"{'='*60}")
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        print(f"Total Requests: {len(self.results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {len(self.results)/total_time:.2f} req/s")
        
        if successful:
            latencies = [r.latency for r in successful]
            prompt_tokens = sum(r.prompt_tokens for r in successful)
            generated_tokens = sum(r.generated_tokens for r in successful)
            
            print(f"\nLatency Statistics:")
            print(f"  Mean: {sum(latencies)/len(latencies):.2f}s")
            print(f"  Min: {min(latencies):.2f}s")
            print(f"  Max: {max(latencies):.2f}s")
            print(f"  Median: {sorted(latencies)[len(latencies)//2]:.2f}s")
            
            print(f"\nToken Statistics:")
            print(f"  Total Prompt Tokens: {prompt_tokens}")
            print(f"  Total Generated Tokens: {generated_tokens}")
            print(f"  Avg Prompt Tokens: {prompt_tokens/len(successful):.1f}")
            print(f"  Avg Generated Tokens: {generated_tokens/len(successful):.1f}")
            print(f"  Token Throughput: {generated_tokens/total_time:.2f} tokens/s")
        
        if failed:
            print(f"\nFailed Requests:")
            for r in failed[:5]:  # 只显示前5个失败
                print(f"  Request {r.request_id}: {r.error}")
            if len(failed) > 5:
                print(f"  ... and {len(failed)-5} more")
        
        print(f"{'='*60}\n")
        
        # 提示查看服务器日志
        print("💡 To verify CPU offloading is working, check the server logs for:")
        print("   - 'offload' or 'swap' messages")
        print("   - 'GPU->CPU' or 'CPU->GPU' transfer messages")
        print("   - Block eviction and loading messages")
        print()

    def save_results(self, output_file: str):
        """保存测试结果到文件"""
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "total_requests": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "results": [
                {
                    "request_id": r.request_id,
                    "prompt_tokens": r.prompt_tokens,
                    "generated_tokens": r.generated_tokens,
                    "latency": r.latency,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to: {output_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="vLLM CPU Offloading Stress Test Client"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="vLLM API server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Total number of requests to send (default: 50)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        choices=["unique", "shared", "mixed"],
        default="unique",
        help="Prompt generation pattern (default: unique)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per request (default: 100)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between batches in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (default: None)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    
    args = parser.parse_args()
    
    client = OffloadTestClient(
        base_url=args.url,
        timeout=args.timeout,
    )
    
    await client.run_batch_test(
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        prompt_pattern=args.pattern,
        max_tokens=args.max_tokens,
        delay_between_batches=args.delay,
    )
    
    if args.output:
        client.save_results(args.output)


if __name__ == "__main__":
    asyncio.run(main())
