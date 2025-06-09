import functools
import math
from typing import List, Set, Dict, Any, Callable
from collections import defaultdict

def statistics(*stats: str):
    """统计修饰器
    
    Args:
        *stats: 统计项，可选值：'SUM', 'AVG', 'VAR', 'RMSE'
    """
    valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
    if not all(stat in valid_stats for stat in stats):
        raise ValueError(f"统计项必须是以下之一：{valid_stats}")
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # 获取原始样本集
            samples = func(*args, **kwargs)
            
            # 收集所有数值型数据
            numbers = []
            
            def collect_numbers(data: Any):
                if isinstance(data, (int, float)):
                    numbers.append(data)
                elif isinstance(data, (list, tuple)):
                    for item in data:
                        collect_numbers(item)
                elif isinstance(data, dict):
                    for value in data.values():
                        collect_numbers(value)
            
            # 遍历所有样本收集数值
            for sample in samples:
                collect_numbers(sample)
            
            if not numbers:
                return {stat: None for stat in stats}
            
            # 计算统计结果
            results = {}
            
            if 'SUM' in stats:
                results['SUM'] = sum(numbers)
            
            if 'AVG' in stats:
                results['AVG'] = sum(numbers) / len(numbers)
            
            if 'VAR' in stats or 'RMSE' in stats:
                mean = sum(numbers) / len(numbers)
                squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
                variance = squared_diff_sum / len(numbers)
                
                if 'VAR' in stats:
                    results['VAR'] = variance
                
                if 'RMSE' in stats:
                    results['RMSE'] = math.sqrt(variance)
            
            return results
        
        return wrapper
    return decorator

# 测试代码
if __name__ == "__main__":
    # 定义测试用的数据结构
    test_structure = {
        "type": "list",
        "items": {
            "type": "dict",
            "properties": {
                "id": {"type": "int"},
                "scores": {
                    "type": "list",
                    "items": {"type": "float"}
                },
                "metadata": {
                    "type": "dict",
                    "properties": {
                        "value": {"type": "float"},
                        "count": {"type": "int"}
                    }
                }
            }
        }
    }
    
    # 测试函数
    def generate_test_samples(**kwargs):
        # 模拟生成样本
        samples = []
        for _ in range(kwargs.get('count', 1)):
            sample = {
                'id': 100,
                'scores': [85.5, 92.3],
                'metadata': {
                    'value': 45.6,
                    'count': 10
                }
            }
            samples.append(sample)
        return samples
    
    # 测试1：使用所有统计项
    @statistics('SUM', 'AVG', 'VAR', 'RMSE')
    def test_all_stats(**kwargs):
        return generate_test_samples(**kwargs)
    
    print("测试1 - 所有统计项：")
    samples = test_all_stats(count=3)
    print("生成的样本：")
    for i, sample in enumerate(samples, 1):
        print(f"样本 {i}:", sample)
    print("\n统计结果：")
    for stat, value in samples.items():
        print(f"{stat}: {value:.4f}")
    
    # 测试2：只使用部分统计项
    @statistics('AVG', 'VAR')
    def test_partial_stats(**kwargs):
        return generate_test_samples(**kwargs)
    
    print("\n测试2 - 部分统计项（AVG, VAR）：")
    samples = test_partial_stats(count=3)
    print("生成的样本：")
    for i, sample in enumerate(samples, 1):
        print(f"样本 {i}:", sample)
    print("\n统计结果：")
    for stat, value in samples.items():
        print(f"{stat}: {value:.4f}")
    
    # 测试3：使用更简单的数据结构
    @statistics('SUM', 'AVG')
    def test_simple_stats(**kwargs):
        return [{'value': 45.6} for _ in range(kwargs.get('count', 1))]
    
    print("\n测试3 - 简单数据结构：")
    samples = test_simple_stats(count=5)
    print("生成的样本：")
    for i, sample in enumerate(samples, 1):
        print(f"样本 {i}:", sample)
    print("\n统计结果：")
    for stat, value in samples.items():
        print(f"{stat}: {value:.4f}")
