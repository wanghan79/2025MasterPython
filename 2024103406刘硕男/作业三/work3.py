import math
import functools
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Tuple, Union

def stats_decorator(stats: List[str] = ['SUM', 'AVG', 'VAR', 'RMSE']):
    # 验证统计项参数
    valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
    if not set(stats).issubset(valid_stats):
        invalid = set(stats) - valid_stats
        raise ValueError(f"无效的统计项: {invalid}. 有效选项: {valid_stats}")
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 调用原始函数获取样本集
            samples = func(*args, **kwargs)
            
            # 2. 递归提取所有数值型数据
            all_numbers = []
            for sample in samples:
                all_numbers.extend(extract_numbers(sample))
            
            # 3. 计算统计量
            stats_result = calculate_statistics(all_numbers, stats)
            
            # 4. 返回原始样本集和统计结果
            return samples, stats_result
        
        return wrapper
    return decorator

def extract_numbers(data: Any) -> List[Union[int, float]]:
    """
    递归提取嵌套数据结构中的所有数值型数据
    
    参数:
        data (Any): 任意嵌套的数据结构
        
    返回:
        List[Union[int, float]]: 提取出的数值列表
    """
    numbers = []
    
    # 基本数值类型
    if isinstance(data, (int, float)):
        numbers.append(data)
    
    # 字符串类型 - 尝试转换为数值
    elif isinstance(data, str):
        try:
            num = float(data) if '.' in data else int(data)
            numbers.append(num)
        except ValueError:
            pass  # 忽略无法转换的字符串
    
    # 字典类型 - 递归处理值
    elif isinstance(data, dict):
        for value in data.values():
            numbers.extend(extract_numbers(value))
    
    # 可迭代类型（列表、元组、集合等）
    elif isinstance(data, Iterable) and not isinstance(data, str):
        for item in data:
            numbers.extend(extract_numbers(item))
    
    return numbers

def calculate_statistics(numbers: List[Union[int, float]], 
                         stats: List[str]) -> Dict[str, float]:
    """
    计算数值列表的统计量
    
    参数:
        numbers (List[Union[int, float]]): 数值列表
        stats (List[str]): 需要计算的统计项
        
    返回:
        Dict[str, float]: 统计结果字典
    """
    n = len(numbers)
    result = {}
    
    # 如果没有数值，所有统计项返回NaN
    if n == 0:
        for stat in stats:
            result[stat] = float('nan')
        return result
    
    # 计算基础统计量
    total = sum(numbers)
    mean = total / n
    
    # 计算需要的统计项
    if 'SUM' in stats:
        result['SUM'] = total
    
    if 'AVG' in stats:
        result['AVG'] = mean
    
    if 'VAR' in stats or 'RMSE' in stats:
        # 计算方差: 总体方差 (分母为n)
        variance = sum((x - mean) ** 2 for x in numbers) / n
        if 'VAR' in stats:
            result['VAR'] = variance
        
        if 'RMSE' in stats:
            # RMSE 作为标准差 (方差的平方根)
            result['RMSE'] = math.sqrt(variance)
    
    return result

# 测试示例
if __name__ == "__main__":
    # 重新定义作业2的样本生成函数并应用修饰器
    @stats_decorator(stats=['SUM', 'AVG', 'VAR', 'RMSE'])
    def generate_samples(**kwargs):
        """简化的样本生成函数用于测试"""
        # 实际实现应使用作业2的完整代码
        return [
            {
                "id": 1,
                "values": [10, 20, 30],
                "nested": {
                    "score": 5.5,
                    "matrix": [[1, 2], [3, 4]]
                }
            },
            {
                "id": 2,
                "values": [40, 50, 60],
                "nested": {
                    "score": 7.5,
                    "matrix": [[5, 6], [7, 8]]
                }
            }
        ]
    
    # 调用修饰后的函数
    samples, stats = generate_samples()
    
    print("生成的样本集:")
    for i, sample in enumerate(samples, 1):
        print(f"样本{i}: {sample}")
    
    print("\n统计结果:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.4f}")
    
    # 测试不同统计项组合
    print("\n测试不同统计项组合:")
    
    @stats_decorator(stats=['SUM', 'AVG'])
    def generate_minimal_samples(**kwargs):
        return [{"a": 10}, {"b": [20, 30]}]
    
    samples2, stats2 = generate_minimal_samples()
    print("统计结果(仅SUM和AVG):", stats2)
    
    # 测试空样本集
    @stats_decorator(stats=['SUM', 'AVG', 'VAR'])
    def generate_empty_samples(**kwargs):
        return []
    
    samples3, stats3 = generate_empty_samples()
    print("\n空样本集统计结果:", stats3)
    
    # 测试包含字符串数值的情况
    @stats_decorator(stats=['SUM', 'AVG'])
    def generate_mixed_samples(**kwargs):
        return [
            {"num": 10, "str_num": "20.5"},
            {"list": ["30", "invalid"]},
            {"nested": {"value": "40"}}
        ]
    
    samples4, stats4 = generate_mixed_samples()
    print("\n混合类型样本统计结果:", stats4)
