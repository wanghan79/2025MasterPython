from openai import OpenAI
import random
import time
from typing import List, Dict, Optional
import logging
from functools import lru_cache
import asyncio

class ModelEndpoint:
    def __init__(self, url: str, api_key: str, model_name: str, weight: float = 1.0, max_concurrent_requests: int = 15):
        self.url = url
        self.api_key = api_key
        self.model_name = model_name
        self.weight = weight
        self.client = OpenAI(api_key=api_key, base_url=url)
        self.last_used = 0
        self.error_count = 0
        self.is_available = True
        self.lock = asyncio.Lock()
        self.consecutive_success = 0
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)  # 用asyncio.Semaphore

class ModelManager:
    def __init__(self, endpoints: List[Dict], cooldown_period: float = 0.1, max_retries: int = 2):
        """
        初始化模型管理器
        
        Args:
            endpoints: 模型端点配置列表，每个配置包含 url, api_key, model_name 和可选的 weight
            cooldown_period: 同一端点两次调用之间的最小间隔（秒）
            max_retries: 最大重试次数
        """
        self.endpoints = [
            ModelEndpoint(
                url=ep['url'],
                api_key=ep['api_key'],
                model_name=ep['model_name'],
                weight=ep.get('weight', 1.0)
            ) for ep in endpoints
        ]
        self.cooldown_period = cooldown_period
        self.max_retries = max_retries
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger('ModelManager')
        
        # 性能优化参数
        self.min_cooldown = 0.1  # 最小冷却时间
        self.max_cooldown = 1.0   # 最大冷却时间
        self.success_threshold = 5  # 连续成功多少次后减少冷却时间
        self.disable_timeout = 10   # 端点禁用恢复时间（秒）

    def _select_endpoint(self) -> Optional[ModelEndpoint]:
        """使用加权随机选择算法选择一个可用的端点"""
        available_endpoints = [ep for ep in self.endpoints if ep.is_available]
        if not available_endpoints:
            return None

        # 优先选择连续成功次数多的端点
        available_endpoints.sort(key=lambda x: (-x.consecutive_success, -x.weight))
        
        # 70%的概率选择最佳端点，30%的概率随机选择其他端点
        if random.random() < 0.7 and available_endpoints[0].consecutive_success > 0:
            return available_endpoints[0]

        total_weight = sum(ep.weight for ep in available_endpoints)
        r = random.uniform(0, total_weight)
        current_weight = 0

        for endpoint in available_endpoints:
            current_weight += endpoint.weight
            if r <= current_weight:
                return endpoint

        return available_endpoints[-1]

    def _is_endpoint_ready(self, endpoint: ModelEndpoint) -> bool:
        """检查端点是否已经冷却完毕"""
        current_time = time.time()
        time_since_last_use = current_time - endpoint.last_used
        
        # 根据连续成功次数动态调整冷却时间
        if endpoint.consecutive_success >= self.success_threshold:
            actual_cooldown = max(self.min_cooldown, 
                                self.cooldown_period * (0.8 ** (endpoint.consecutive_success - self.success_threshold)))
        else:
            actual_cooldown = self.cooldown_period
            
        return time_since_last_use >= actual_cooldown

    async def get_completion(self, messages: List[Dict], max_tokens: Optional[int] = None, **kwargs) -> Optional[Dict]:
        """
        获取模型响应，包含自动重试和负载均衡
        
        Args:
            messages: 消息列表
            max_tokens: 最大tokens数
            **kwargs: 传递给OpenAI API的其他参数
        
        Returns:
            模型响应或None（如果所有尝试都失败）
        """
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            endpoint = self._select_endpoint()
            if not endpoint:
                self.logger.error("没有可用的模型端点")
                return None

            try:
                async with endpoint.request_semaphore:
                    start_time = time.time()
                    response = await asyncio.to_thread(
                        endpoint.client.chat.completions.create,
                        model=endpoint.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                    
                    async with endpoint.lock:
                        endpoint.last_used = time.time()
                        endpoint.error_count = 0
                        endpoint.consecutive_success += 1
                    
                    # 记录响应时间
                    response_time = time.time() - start_time
                    self.logger.debug(f"端点 {endpoint.url} 响应时间: {response_time:.2f}秒")
                    
                    return response

            except Exception as e:
                last_error = str(e)
                self.logger.error(f"端点 {endpoint.url} 调用失败: {last_error}")
                async with endpoint.lock:
                    endpoint.error_count += 1
                    endpoint.consecutive_success = 0
                    
                    # 如果错误次数过多，暂时禁用该端点
                    if endpoint.error_count >= 3:
                        endpoint.is_available = False
                        self.logger.warning(f"端点 {endpoint.url} 暂时禁用")
                        
                        # 启动一个定时器，在指定时间后重新启用该端点
                        async def enable_endpoint():
                            await asyncio.sleep(self.disable_timeout)
                            async with endpoint.lock:
                                endpoint.is_available = True
                                endpoint.error_count = 0
                                self.logger.info(f"端点 {endpoint.url} 重新启用")
                        
                        asyncio.create_task(enable_endpoint())

            attempt += 1
            if attempt < self.max_retries:
                await asyncio.sleep(0.2)  # 减少重试等待时间

        self.logger.error(f"所有重试都失败了，最后的错误: {last_error}")
        return None
    
    async def call_llm(self, prompt, system_prompt=None, history_messages=None, max_tokens=None):
        # 构造 messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        # model_manager.get_completion 是同步方法，需要用 asyncio.to_thread 包装
        response = await self.get_completion(messages, max_tokens=max_tokens)
        return response

@lru_cache(maxsize=1)
def get_model_manager(model, endpoints=None):
    if MODEL_ENDPOINTS is None and endpoints is None:
        raise ValueError("MODEL_ENDPOINTS 未定义")
    else:
        if model == "Qwen2.5-72B-Instruct":
            endpoints = MODEL_ENDPOINTS["Qwen2.5-72B-Instruct"] or endpoints
        elif model == "MiniCPM-V-2_6-int4":
            endpoints = MODEL_ENDPOINTS["MiniCPM-V-2_6-int4"] or endpoints
        else:
            raise ValueError(f"不支持的模型: {model}")
    return ModelManager(endpoints=endpoints, max_retries=10000)