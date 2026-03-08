import json
import time
import asyncio
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import random
from openai import AsyncOpenAI
from datetime import datetime
import tiktoken

# ============== 全局随机单词环形池 ==============
# 预生成10000个随机单词的池子，避免每次请求时重复生成
WORD_POOL_SIZE = 10000
COMMON_WORDS_FOR_POOL = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
]

def _init_word_pool() -> Tuple[List[str], str]:
    """初始化全局单词池，返回单词列表和预拼接的字符串"""
    word_list = [random.choice(COMMON_WORDS_FOR_POOL) for _ in range(WORD_POOL_SIZE)]
    word_string = " ".join(word_list)
    return word_list, word_string

# 全局单词池 - 在模块加载时初始化
GLOBAL_WORD_LIST, GLOBAL_WORD_STRING = _init_word_pool()
# 预计算每个单词在字符串中的起始位置（用于快速定位）
GLOBAL_WORD_POSITIONS = []
_pos = 0
for i, word in enumerate(GLOBAL_WORD_LIST):
    GLOBAL_WORD_POSITIONS.append(_pos)
    _pos += len(word) + 1  # +1 for space

print(f"[初始化] 全局单词池已创建: {WORD_POOL_SIZE} 个单词, 总长度: {len(GLOBAL_WORD_STRING)} 字符")
# ============================================

@dataclass
class RequestMetrics:
    """单个请求的性能指标"""
    request_id: str
    chat_id: int
    turn: int
    is_single_turn: bool
    prompt_tokens: int
    completion_tokens: int
    ttft: float  # Time To First Token (秒)
    tpot: float  # Time Per Output Token (秒)
    total_time: float  # 总时间 (秒)
    timestamp: float  # 请求发送时间
    success: bool
    error_msg: Optional[str] = None
    is_prefetch: bool = False  # 是否是prefetch请求

@dataclass
class WorkloadStats:
    """整体workload统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    
    ttft_list: List[float] = field(default_factory=list)
    tpot_list: List[float] = field(default_factory=list)
    total_time_list: List[float] = field(default_factory=list)
    
    single_turn_count: int = 0
    multi_turn_count: int = 0
    
    # Prefetch 统计（不参与主要指标计算）
    prefetch_total: int = 0
    prefetch_successful: int = 0
    prefetch_failed: int = 0
    prefetch_ttft_list: List[float] = field(default_factory=list)
    prefetch_total_time_list: List[float] = field(default_factory=list)
    
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def add_metric(self, metric: RequestMetrics):
        """添加一个请求的指标"""
        # Prefetch 请求单独统计，不参与主要指标计算
        if metric.is_prefetch:
            self.prefetch_total += 1
            if metric.success:
                self.prefetch_successful += 1
                self.prefetch_ttft_list.append(metric.ttft)
                self.prefetch_total_time_list.append(metric.total_time)
            else:
                self.prefetch_failed += 1
            return  # 不参与主要指标统计
        
        self.total_requests += 1
        if metric.success:
            self.successful_requests += 1
            self.total_prompt_tokens += metric.prompt_tokens
            self.total_completion_tokens += metric.completion_tokens
            self.ttft_list.append(metric.ttft)
            self.tpot_list.append(metric.tpot)
            self.total_time_list.append(metric.total_time)
            
            if metric.is_single_turn:
                self.single_turn_count += 1
            else:
                self.multi_turn_count += 1
        else:
            self.failed_requests += 1
    
    def print_report(self):
        """打印性能报告"""
        print("\n" + "="*80)
        print("性能测试报告")
        print("="*80)
        
        if self.start_time and self.end_time:
            actual_duration = self.end_time - self.start_time
            print(f"\n测试时长: {actual_duration:.2f} 秒")
            actual_qps = self.total_requests / actual_duration
            print(f"实际QPS: {actual_qps:.2f} req/s")
        
        print(f"\n请求统计:")
        print(f"  总请求数: {self.total_requests}")
        print(f"  成功请求: {self.successful_requests}")
        print(f"  失败请求: {self.failed_requests}")
        print(f"  成功率: {self.successful_requests/self.total_requests*100:.2f}%" if self.total_requests > 0 else "  成功率: N/A")
        print(f"  单轮对话: {self.single_turn_count}")
        print(f"  多轮对话: {self.multi_turn_count}")
        
        # Prefetch 统计
        if self.prefetch_total > 0:
            print(f"\nPrefetch 请求统计 (不参与性能指标计算):")
            print(f"  Prefetch 总数: {self.prefetch_total}")
            print(f"  Prefetch 成功: {self.prefetch_successful}")
            print(f"  Prefetch 失败: {self.prefetch_failed}")
            print(f"  Prefetch 成功率: {self.prefetch_successful/self.prefetch_total*100:.2f}%")
            if self.prefetch_ttft_list:
                print(f"  Prefetch 平均TTFT: {np.mean(self.prefetch_ttft_list)*1000:.2f} ms")
                print(f"  Prefetch 平均总时间: {np.mean(self.prefetch_total_time_list)*1000:.2f} ms")
        
        print(f"\nToken统计:")
        print(f"  总prompt tokens: {self.total_prompt_tokens}")
        print(f"  总completion tokens: {self.total_completion_tokens}")
        print(f"  平均prompt tokens: {self.total_prompt_tokens/self.successful_requests:.2f}")
        print(f"  平均completion tokens: {self.total_completion_tokens/self.successful_requests:.2f}")
        
        if self.ttft_list:
            print(f"\nTTFT (Time To First Token) 统计:")
            print(f"  平均值: {np.mean(self.ttft_list)*1000:.2f} ms")
            print(f"  标准差: {np.std(self.ttft_list)*1000:.2f} ms")
            print(f"  中位数: {np.median(self.ttft_list)*1000:.2f} ms")
            print(f"  P50: {np.percentile(self.ttft_list, 50)*1000:.2f} ms")
            print(f"  P90: {np.percentile(self.ttft_list, 90)*1000:.2f} ms")
            print(f"  P95: {np.percentile(self.ttft_list, 95)*1000:.2f} ms")
            print(f"  P99: {np.percentile(self.ttft_list, 99)*1000:.2f} ms")
            print(f"  最小值: {np.min(self.ttft_list)*1000:.2f} ms")
            print(f"  最大值: {np.max(self.ttft_list)*1000:.2f} ms")
        
        if self.tpot_list:
            print(f"\nTPOT (Time Per Output Token) 统计:")
            print(f"  平均值: {np.mean(self.tpot_list)*1000:.2f} ms/token")
            print(f"  标准差: {np.std(self.tpot_list)*1000:.2f} ms/token")
            print(f"  中位数: {np.median(self.tpot_list)*1000:.2f} ms/token")
            print(f"  P50: {np.percentile(self.tpot_list, 50)*1000:.2f} ms/token")
            print(f"  P90: {np.percentile(self.tpot_list, 90)*1000:.2f} ms/token")
            print(f"  P95: {np.percentile(self.tpot_list, 95)*1000:.2f} ms/token")
            print(f"  P99: {np.percentile(self.tpot_list, 99)*1000:.2f} ms/token")
        
        if self.total_time_list:
            print(f"\n总响应时间统计:")
            print(f"  平均值: {np.mean(self.total_time_list):.3f} 秒")
            print(f"  标准差: {np.std(self.total_time_list):.3f} 秒")
            print(f"  中位数: {np.median(self.total_time_list):.3f} 秒")
            print(f"  P90: {np.percentile(self.total_time_list, 90):.3f} 秒")
            print(f"  P95: {np.percentile(self.total_time_list, 95):.3f} 秒")
            print(f"  P99: {np.percentile(self.total_time_list, 99):.3f} 秒")
        
        print("="*80 + "\n")
    
    def save_to_file(self, output_file: str):
        """保存详细结果到JSON文件"""
        data = {
            "summary": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": self.successful_requests/self.total_requests*100 if self.total_requests > 0 else 0,
                "single_turn_count": self.single_turn_count,
                "multi_turn_count": self.multi_turn_count,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "avg_prompt_tokens": self.total_prompt_tokens/self.successful_requests if self.successful_requests > 0 else 0,
                "avg_completion_tokens": self.total_completion_tokens/self.successful_requests if self.successful_requests > 0 else 0,
                "duration_seconds": self.end_time - self.start_time if self.start_time and self.end_time else 0,
                "actual_qps": self.total_requests / (self.end_time - self.start_time) if self.start_time and self.end_time else 0,
            },
            "ttft": {
                "mean_ms": np.mean(self.ttft_list)*1000 if self.ttft_list else 0,
                "std_ms": np.std(self.ttft_list)*1000 if self.ttft_list else 0,
                "median_ms": np.median(self.ttft_list)*1000 if self.ttft_list else 0,
                "p50_ms": np.percentile(self.ttft_list, 50)*1000 if self.ttft_list else 0,
                "p90_ms": np.percentile(self.ttft_list, 90)*1000 if self.ttft_list else 0,
                "p95_ms": np.percentile(self.ttft_list, 95)*1000 if self.ttft_list else 0,
                "p99_ms": np.percentile(self.ttft_list, 99)*1000 if self.ttft_list else 0,
            },
            "tpot": {
                "mean_ms": np.mean(self.tpot_list)*1000 if self.tpot_list else 0,
                "std_ms": np.std(self.tpot_list)*1000 if self.tpot_list else 0,
                "median_ms": np.median(self.tpot_list)*1000 if self.tpot_list else 0,
                "p50_ms": np.percentile(self.tpot_list, 50)*1000 if self.tpot_list else 0,
                "p90_ms": np.percentile(self.tpot_list, 90)*1000 if self.tpot_list else 0,
                "p95_ms": np.percentile(self.tpot_list, 95)*1000 if self.tpot_list else 0,
                "p99_ms": np.percentile(self.tpot_list, 99)*1000 if self.tpot_list else 0,
            },
            "prefetch": {
                "total": self.prefetch_total,
                "successful": self.prefetch_successful,
                "failed": self.prefetch_failed,
                "success_rate": self.prefetch_successful/self.prefetch_total*100 if self.prefetch_total > 0 else 0,
                "mean_ttft_ms": np.mean(self.prefetch_ttft_list)*1000 if self.prefetch_ttft_list else 0,
                "mean_total_time_ms": np.mean(self.prefetch_total_time_list)*1000 if self.prefetch_total_time_list else 0,
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"详细结果已保存到: {output_file}")


class WorkloadGenerator:
    """Workload生成器"""
    
    # 常用英语单词列表用于填充
    COMMON_WORDS = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
    ]
    
    def __init__(self, trace_file: str, api_base: str, api_key: str = "EMPTY", model: str = "qwen",
                 prefetch_seconds: float = 0.0):
        self.trace_file = trace_file
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.prefetch_seconds = prefetch_seconds  # Prefetch 提前量（秒），0表示禁用
        
        # 读取trace数据
        self.records = []
        self.chat_dict = {}
        self.children_dict = defaultdict(list)
        self.single_turn_conversations = []
        self.multi_turn_conversations = []
        
        self._load_trace()
        self._analyze_conversations()
        
        # OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        
        # 统计
        self.stats = WorkloadStats()
        
        # 初始化tokenizer用于精确计算token数量
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
            print("警告: 无法加载tiktoken，将使用近似的token计数方法")
        
        # Timeout 相关配置
        self.timeout: Optional[float] = None  # 超时时间（秒）
        self.test_start_time: Optional[float] = None  # 测试开始时间
        
        if self.prefetch_seconds > 0:
            print(f"[Prefetch] 已启用，提前量: {self.prefetch_seconds} 秒")
    
    def _is_timeout(self) -> bool:
        """检查是否已经超时"""
        if self.timeout is None or self.test_start_time is None:
            return False
        return time.time() - self.test_start_time >= self.timeout
    
    def _load_trace(self):
        """加载trace文件"""
        print(f"正在加载trace文件: {self.trace_file}")
        with open(self.trace_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        self.records.append(record)
                        self.chat_dict[record['chat_id']] = record
                    except json.JSONDecodeError:
                        continue
        
        # 构建父子关系
        for record in self.records:
            parent_id = record['parent_chat_id']
            if parent_id != -1:
                self.children_dict[parent_id].append(record['chat_id'])
        
        print(f"加载完成，共 {len(self.records)} 条记录")
    
    def _analyze_conversations(self):
        """分析单轮和多轮对话"""
        for record in self.records:
            chat_id = record['chat_id']
            parent_id = record['parent_chat_id']
            
            if parent_id == -1:
                if len(self.children_dict[chat_id]) == 0:
                    self.single_turn_conversations.append(chat_id)
                else:
                    self.multi_turn_conversations.append(chat_id)
        
        print(f"单轮对话: {len(self.single_turn_conversations)}, 多轮对话: {len(self.multi_turn_conversations)}")
    
    def _get_conversation_chain(self, root_id: int) -> List[Dict]:
        """获取完整的对话链（BFS遍历）"""
        conversation_chain = []
        queue = [root_id]
        visited = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id in self.chat_dict:
                conversation_chain.append(self.chat_dict[current_id])
                queue.extend(self.children_dict[current_id])
        
        # 按turn排序
        conversation_chain.sort(key=lambda x: x['turn'])
        return conversation_chain
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # 近似计算：英文约4个字符=1个token
            return len(text) // 4
    
    def _generate_text_with_tokens(self, target_tokens: int) -> str:
        """
        生成具有指定token数量的文本
        
        使用全局环形单词池，从随机起点截取连续的一段文本，性能远优于逐词生成。
        """
        if target_tokens <= 0:
            return ""
        
        # 粗略估算需要多少单词（英文平均1.3个token/word）
        estimated_words = int(target_tokens * 1.5)
        
        # 确保不超过池子大小
        if estimated_words > WORD_POOL_SIZE:
            estimated_words = WORD_POOL_SIZE
        
        # 从池子中随机选一个起点
        start_idx = random.randint(0, WORD_POOL_SIZE - 1)
        
        # 从全局单词池中环形截取
        if start_idx + estimated_words <= WORD_POOL_SIZE:
            # 不需要绕回，直接切片
            text = " ".join(GLOBAL_WORD_LIST[start_idx:start_idx + estimated_words])
        else:
            # 需要绕回，拼接两段
            first_part = GLOBAL_WORD_LIST[start_idx:]
            remaining = estimated_words - len(first_part)
            second_part = GLOBAL_WORD_LIST[:remaining]
            text = " ".join(first_part + second_part)
        
        current_tokens = self._count_tokens(text)
        
        # 微调：如果token数不够，增加单词数量（仍使用环形切片）
        words_count = estimated_words
        while current_tokens < target_tokens and words_count < WORD_POOL_SIZE:
            words_count += 10  # 每次增加10个单词
            if start_idx + words_count <= WORD_POOL_SIZE:
                text = " ".join(GLOBAL_WORD_LIST[start_idx:start_idx + words_count])
            else:
                first_part = GLOBAL_WORD_LIST[start_idx:]
                remaining = words_count - len(first_part)
                if remaining > WORD_POOL_SIZE:
                    remaining = WORD_POOL_SIZE
                second_part = GLOBAL_WORD_LIST[:remaining]
                text = " ".join(first_part + second_part)
            current_tokens = self._count_tokens(text)
        
        # 如果超出太多，尝试删减
        while current_tokens > target_tokens * 1.1 and words_count > 1:
            words_count -= 5
            if words_count < 1:
                words_count = 1
            if start_idx + words_count <= WORD_POOL_SIZE:
                text = " ".join(GLOBAL_WORD_LIST[start_idx:start_idx + words_count])
            else:
                first_part = GLOBAL_WORD_LIST[start_idx:]
                remaining = words_count - len(first_part)
                if remaining < 0:
                    remaining = 0
                second_part = GLOBAL_WORD_LIST[:remaining]
                text = " ".join(first_part + second_part)
            current_tokens = self._count_tokens(text)
        
        return text
    
    def sample_workload(self, 
                       num_single_turn: int,
                       num_multi_turn: int,
                       max_turns: Optional[int] = None) -> List[Tuple[str, List[Dict]]]:
        """
        选取workload（只使用头部的N个多轮对话，不再随机采样）
        
        Args:
            num_single_turn: 单轮对话数量（现在会被忽略）
            num_multi_turn: 多轮对话数量（取前N个）
            max_turns: 多轮对话最大轮数限制（None表示不限制）
        
        Returns:
            List of (conversation_type, conversation_chain)
        """
        workload = []
        
        # 只使用头部的N个多轮对话（不再采样单轮对话）
        num_to_take = min(num_multi_turn, len(self.multi_turn_conversations))
        selected_multi = self.multi_turn_conversations[:num_to_take]
        
        for root_id in selected_multi:
            chain = self._get_conversation_chain(root_id)
            if max_turns:
                chain = chain[:max_turns]
            workload.append(("multi", chain))
        
        print(f"\n已选取workload:")
        print(f"  多轮对话: {len(selected_multi)} (头部)")
        print(f"  总对话数: {len(workload)}")
        print(f"  总请求数: {sum(len(chain) for _, chain in workload)}")
        
        return workload
    
    async def _send_request(self, 
                           messages: List[Dict[str, str]], 
                           chat_id: int,
                           turn: int,
                           is_single_turn: bool,
                           target_output_tokens: int) -> Tuple[RequestMetrics, str]:
        """
        发送单个请求并测量性能指标
        
        Returns:
            (RequestMetrics, response_text): 返回指标和LLM的完整响应文本
        """
        request_id = f"{chat_id}_turn{turn}"
        
        # 精确计算prompt tokens
        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt_tokens = self._count_tokens(prompt_text)
        
        # 日志：发送请求
        print(f"[发送请求] {request_id} | Input tokens: {prompt_tokens}, Target output tokens: {target_output_tokens}")
        
        start_time = time.time()
        first_token_time = None
        completion_tokens = 0
        success = False
        error_msg = None
        response_text = ""
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=target_output_tokens,
                stream=True,
                # 将 ignore_eos 放入 extra_body 字典中
                extra_body={
                    "ignore_eos": True
                }
            )
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        content = chunk.choices[0].delta.content
                        response_text += content
            
            # 计算实际生成的token数量
            if response_text:
                completion_tokens = self._count_tokens(response_text)
            
            success = True
            
        except Exception as e:
            error_msg = str(e)
            print(f"请求失败 [{request_id}]: {error_msg}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算TTFT和TPOT
        if first_token_time and success:
            ttft = first_token_time - start_time
            if completion_tokens > 1:
                tpot = (end_time - first_token_time) / (completion_tokens - 1)
            else:
                tpot = 0
        else:
            ttft = 0
            tpot = 0
        
        metrics = RequestMetrics(
            request_id=request_id,
            chat_id=chat_id,
            turn=turn,
            is_single_turn=is_single_turn,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft=ttft,
            tpot=tpot,
            total_time=total_time,
            timestamp=start_time,
            success=success,
            error_msg=error_msg
        )
        
        return metrics, response_text
    
    async def _send_prefetch_request(self, 
                                     messages: List[Dict[str, str]], 
                                     chat_id: int,
                                     turn: int) -> RequestMetrics:
        """
        发送 prefetch 请求（用于预热 KV cache）
        
        Prefetch 请求特点：
        - max_tokens=1，只触发 prefill，生成尽量少的 token
        - 不参与 TTFT/TPOT 统计
        - 用于提前将 KV cache 从 CPU 加载到 GPU
        
        Returns:
            RequestMetrics: prefetch 请求的指标（is_prefetch=True）
        """
        request_id = f"{chat_id}_turn{turn}_prefetch"
        
        # 计算 prompt tokens
        prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt_tokens = self._count_tokens(prompt_text)
        
        # 日志：发送 prefetch 请求
        print(f"[Prefetch发送] {request_id} | Input tokens: {prompt_tokens}, max_tokens: 1")
        
        start_time = time.time()
        first_token_time = None
        completion_tokens = 0
        success = False
        error_msg = None
        response_text = ""
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1,  # 只生成1个token，主要是触发prefill
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        content = chunk.choices[0].delta.content
                        response_text += content
            
            # 计算实际生成的 token 数量
            if response_text:
                completion_tokens = self._count_tokens(response_text)
            
            success = True
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Prefetch失败] {request_id}: {error_msg}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算 TTFT
        if first_token_time and success:
            ttft = first_token_time - start_time
        else:
            ttft = 0
        
        metrics = RequestMetrics(
            request_id=request_id,
            chat_id=chat_id,
            turn=turn,
            is_single_turn=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft=ttft,
            tpot=0,  # Prefetch 不计算 TPOT
            total_time=total_time,
            timestamp=start_time,
            success=success,
            error_msg=error_msg,
            is_prefetch=True  # 标记为 prefetch 请求
        )
        
        # 打印完成日志
        status = "✓" if success else "✗"
        print(f"[Prefetch完成] {status} {request_id} | "
              f"TTFT: {ttft*1000:.2f}ms, Total: {total_time*1000:.2f}ms | "
              f"Input: {prompt_tokens} tokens")
        
        return metrics
    
    async def _process_conversation(self, 
                                   conv_type: str, 
                                   chain: List[Dict],
                                   scheduled_timestamp: float):
        """
        处理一个对话（单轮或多轮）
        
        Args:
            conv_type: 对话类型 ("single" or "multi")
            chain: 对话链
            scheduled_timestamp: 这个对话的调度时间戳
        """
        messages = []
        prefetch_tasks = []  # 跟踪 prefetch 任务
        
        for i, record in enumerate(chain):
            # 检查是否已超时（超时后不再发送新请求）
            if self._is_timeout():
                request_id = f"{record['chat_id']}_turn{record['turn']}"
                elapsed = time.time() - self.test_start_time if self.test_start_time else 0
                print(f"[超时跳过] {request_id} | 已运行 {elapsed:.2f}s，超过 timeout {self.timeout}s，跳过剩余 {len(chain) - i} 个请求")
                break
            
            # 对于多轮对话的后续轮次（i > 0），提前发送 prefetch 请求
            if i > 0 and self.prefetch_seconds > 0 and conv_type == "multi":
                # 计算 prefetch 发送时间
                prefetch_time = scheduled_timestamp - self.prefetch_seconds
                current_time = time.time()
                prefetch_wait_time = prefetch_time - current_time
                
                if prefetch_wait_time > 0:
                    # 等待到 prefetch 时间
                    request_id = f"{record['chat_id']}_turn{record['turn']}"
                    print(f"[Prefetch等待] {request_id} | 等待 {prefetch_wait_time:.3f} 秒后发送 prefetch")
                    await asyncio.sleep(prefetch_wait_time)
                    
                    # 等待后检查是否超时
                    if self._is_timeout():
                        elapsed = time.time() - self.test_start_time if self.test_start_time else 0
                        print(f"[超时跳过] {request_id} | 已运行 {elapsed:.2f}s，超过 timeout {self.timeout}s，跳过剩余 {len(chain) - i} 个请求")
                        break
                
                # 发送 prefetch 请求（使用当前已有的对话历史，不包含当前轮的新用户消息）
                # Prefetch 请求不阻塞，使用 create_task
                if messages:  # 只有在有历史消息时才发送 prefetch
                    prefetch_task = asyncio.create_task(
                        self._send_prefetch_request(
                            messages=messages.copy(),
                            chat_id=record['chat_id'],
                            turn=record['turn']
                        )
                    )
                    prefetch_tasks.append(prefetch_task)
            
            # 等待到该轮次的预定发送时间
            current_time = time.time()
            wait_time = scheduled_timestamp - current_time
            
            if wait_time > 0:
                request_id = f"{record['chat_id']}_turn{record['turn']}"
                print(f"[等待] {request_id} | 等待 {wait_time:.3f} 秒后发送 (scheduled: {scheduled_timestamp:.3f}, current: {current_time:.3f})")
                await asyncio.sleep(wait_time)
                
                # 等待后再次检查是否超时
                if self._is_timeout():
                    elapsed = time.time() - self.test_start_time if self.test_start_time else 0
                    print(f"[超时跳过] {request_id} | 已运行 {elapsed:.2f}s，超过 timeout {self.timeout}s，跳过剩余 {len(chain) - i} 个请求")
                    break
            
            # 计算需要的input tokens数量
            target_input_tokens = record['input_length']
            
            # 对于多轮对话，需要包含之前的完整对话历史
            # 计算历史对话已经占用的tokens
            history_tokens = 0
            if messages:
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                history_tokens = self._count_tokens(history_text)
            
            # 计算当前轮次新增的用户输入应该有多少tokens
            # 确保总的input tokens等于target_input_tokens
            new_user_tokens = max(10, target_input_tokens - history_tokens)
            
            # 生成具有指定token数量的用户消息
            user_message = self._generate_text_with_tokens(new_user_tokens)
            messages.append({"role": "user", "content": user_message})
            
            # 发送请求，并获取真实的LLM响应
            metric, response_text = await self._send_request(
                messages=messages.copy(),
                chat_id=record['chat_id'],
                turn=record['turn'],
                is_single_turn=(conv_type == "single"),
                target_output_tokens=record['output_length']
            )
            
            self.stats.add_metric(metric)
            
            # 将LLM的真实响应添加到messages中，用于下一轮的context
            if metric.success and response_text:
                messages.append({"role": "assistant", "content": response_text})
            else:
                # 如果失败，也添加一个占位符以保持对话结构
                placeholder = self._generate_text_with_tokens(record['output_length'])
                messages.append({"role": "assistant", "content": placeholder})
            
            # 打印完成日志
            status = "✓" if metric.success else "✗"
            print(f"[完成] {status} {metric.request_id} | "
                  f"TTFT: {metric.ttft*1000:.2f}ms, TPOT: {metric.tpot*1000:.2f}ms, "
                  f"Total: {metric.total_time:.3f}s | "
                  f"Input: {metric.prompt_tokens} tokens, Output: {metric.completion_tokens} tokens")
            
            # 如果是多轮对话，为下一轮更新scheduled_timestamp
            if i < len(chain) - 1:
                # 下一轮的时间戳（已经过缩放）
                next_record = chain[i + 1]
                scheduled_timestamp = next_record.get('scaled_timestamp', scheduled_timestamp)
        
        # 等待所有 prefetch 任务完成并收集统计
        if prefetch_tasks:
            prefetch_results = await asyncio.gather(*prefetch_tasks, return_exceptions=True)
            for result in prefetch_results:
                if isinstance(result, RequestMetrics):
                    self.stats.add_metric(result)
                elif isinstance(result, Exception):
                    print(f"[Prefetch异常] {result}")
    
    def _scale_timestamps(self, workload: List[Tuple[str, List[Dict]]], qps: float):
        """
        按照QPS约束等比例缩放原始时间戳，确保第一个请求从时间0开始
        
        Args:
            workload: workload列表
            qps: 目标QPS（请求级别，非对话级别）
        
        Returns:
            缩放后的workload，每个record增加scaled_timestamp字段
        """
        # 收集所有请求及其原始时间戳
        all_requests = []
        for conv_type, chain in workload:
            for record in chain:
                all_requests.append(record)
        
        if not all_requests:
            return workload
        
        # 找出原始trace的时间范围
        original_timestamps = [r['timestamp'] for r in all_requests]
        min_timestamp = min(original_timestamps)
        max_timestamp = max(original_timestamps)
        original_duration = max_timestamp - min_timestamp
        
        # 计算目标持续时间
        total_requests = len(all_requests)
        target_duration = total_requests / qps
        
        # 计算缩放因子
        if original_duration > 0:
            scale_factor = target_duration / original_duration
        else:
            scale_factor = 1.0
        
        print(f"\n时间戳缩放信息:")
        print(f"  原始时间范围: {original_duration:.2f} 秒 (从 {min_timestamp} 到 {max_timestamp})")
        print(f"  目标持续时间: {target_duration:.2f} 秒 (从 0 开始)")
        print(f"  缩放因子: {scale_factor:.4f}")
        print(f"  总请求数: {total_requests}")
        print(f"  目标QPS: {qps:.2f} req/s")
        
        # 为每个record添加缩放后的时间戳
        scaled_workload = []
        for conv_type, chain in workload:
            scaled_chain = []
            for record in chain:
                # 创建record的副本
                scaled_record = record.copy()
                # 计算相对于最小时间戳的偏移
                relative_time = record['timestamp'] - min_timestamp
                # 应用缩放因子
                scaled_relative_time = relative_time * scale_factor
                # 保存缩放后的时间戳（从0开始）
                scaled_record['scaled_timestamp'] = scaled_relative_time
                scaled_chain.append(scaled_record)
            scaled_workload.append((conv_type, scaled_chain))
        
        return scaled_workload
    
    async def run_workload(self,
                          workload: List[Tuple[str, List[Dict]]],
                          qps: float,
                          duration: Optional[float] = None,
                          timeout: Optional[float] = None):
        """
        运行workload测试
        
        Args:
            workload: 采样的workload
            qps: 每秒请求数（请求级别）
            duration: 测试持续时间（秒），None表示执行完所有workload
            timeout: 测试超时时间（秒），超时后不再发送新请求，None表示不设置超时
        """
        print(f"\n开始运行workload测试...")
        print(f"目标QPS: {qps} req/s")
        if duration:
            print(f"测试时长: {duration} 秒")
        else:
            print(f"测试时长: 执行完所有workload")
        if timeout:
            print(f"超时设置: {timeout} 秒 (超时后不再发送新请求)")
        print("-" * 80)
        
        # 设置 timeout
        self.timeout = timeout
        
        # 缩放时间戳
        scaled_workload = self._scale_timestamps(workload, qps)
        
        self.stats.start_time = time.time()
        self.test_start_time = self.stats.start_time  # 同时设置实例变量用于超时检查
        test_start_time = self.stats.start_time
        
        print(f"\n测试开始时间: {test_start_time:.3f}")
        print(f"所有任务已调度，开始执行...\n")
        
        tasks = []
        
        # 为每个对话创建任务
        conversation_idx = 0
        for conv_type, chain in scaled_workload:
            if not chain:
                continue
            
            # 使用第一个请求的缩放时间戳作为对话开始时间
            first_request_timestamp = test_start_time + chain[0]['scaled_timestamp']
            
            # 如果设置了duration，检查是否超时
            if duration:
                if chain[0]['scaled_timestamp'] >= duration:
                    continue
            
            # 为chain中的每个请求更新绝对时间戳
            for record in chain:
                record['scaled_timestamp'] = test_start_time + record['scaled_timestamp']
            
            # 日志：调度对话
            conversation_idx += 1
            print(f"[调度对话 #{conversation_idx}] chat_id: {chain[0]['chat_id']}, "
                  f"轮数: {len(chain)}, 第一个请求时间: {chain[0]['scaled_timestamp'] - test_start_time:.3f}s")
            
            # 创建对话处理任务
            task = asyncio.create_task(
                self._process_conversation(conv_type, chain, first_request_timestamp)
            )
            tasks.append(task)
        
        # 等待所有任务完成
        print(f"\n{'='*80}")
        print(f"所有对话已调度 (共 {len(tasks)} 个对话)，等待所有请求完成...")
        print(f"{'='*80}\n")
        await asyncio.gather(*tasks)
        
        self.stats.end_time = time.time()
        print(f"\n所有请求已完成！测试结束时间: {self.stats.end_time:.3f}")
        
        # 打印报告
        self.stats.print_report()


async def main():
    parser = argparse.ArgumentParser(description='Workload Generator for LLM Testing')
    
    # 输入文件
    parser.add_argument('--trace-file', type=str, required=True,
                       help='Path to trace JSONL file')
    
    # vLLM服务器配置
    parser.add_argument('--api-base', type=str, default='http://localhost:8000/v1',
                       help='vLLM API base URL')
    parser.add_argument('--api-key', type=str, default='EMPTY',
                       help='API key')
    parser.add_argument('--model', type=str, default='qwen',
                       help='Model name')
    
    # Workload配置
    parser.add_argument('--num-single-turn', type=int, default=50,
                       help='Number of single-turn conversations to sample')
    parser.add_argument('--num-multi-turn', type=int, default=50,
                       help='Number of multi-turn conversations to sample')
    parser.add_argument('--max-turns', type=int, default=None,
                       help='Maximum turns per multi-turn conversation')
    
    # 测试配置
    parser.add_argument('--qps', type=float, required=True,
                       help='Target QPS (requests per second, not conversations per second)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Test duration in seconds (None = run all workload)')
    parser.add_argument('--timeout', type=float, default=None,
                       help='Test timeout in seconds. After timeout, no new requests will be sent, '
                            'but ongoing requests will be completed. (None = no timeout)')
    
    # Prefetch 配置
    parser.add_argument('--prefetch-seconds', type=float, default=0.0,
                       help='Prefetch lead time in seconds (0 = disabled). '
                            'For multi-turn conversations, a prefetch request will be sent '
                            'this many seconds before the actual request to pre-warm the KV cache.')
    
    # 输出配置
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for detailed results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()

    # 设置随机种子
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    
    # 创建workload生成器
    generator = WorkloadGenerator(
        trace_file=args.trace_file,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        prefetch_seconds=args.prefetch_seconds
    )
    
    # 采样workload
    workload = generator.sample_workload(
        num_single_turn=args.num_single_turn,
        num_multi_turn=args.num_multi_turn,
        max_turns=args.max_turns
    )
    
    # 运行测试
    await generator.run_workload(
        workload=workload,
        qps=args.qps,
        duration=args.duration,
        timeout=args.timeout
    )
    
    # 保存结果
    if args.output:
        generator.stats.save_to_file(args.output)


if __name__ == "__main__":
    asyncio.run(main())
