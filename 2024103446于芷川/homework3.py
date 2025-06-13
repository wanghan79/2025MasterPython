import math
import functools
from typing import Any, Dict, List, Tuple, Union, Callable, Set
from collections import defaultdict

def stats_decorator(*operations):
    """
    带参数的装饰器，用于统计数据中数值型数据的特征
    
    参数:
        operations: 统计操作的集合，可以是 'SUM', 'AVG', 'VAR', 'RMSE' 中的任意组合
    
    返回:
        装饰后的函数
    """
    # 验证操作参数
    valid_operations = {'SUM', 'AVG', 'VAR', 'RMSE'}
    for op in operations:
        if op not in valid_operations:
            raise ValueError(f"不支持的操作: {op}，支持的操作有: {valid_operations}")
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 执行原函数获取数据
            result = func(*args, **kwargs)
            
            # 分析数据
            stats = analyze(result, set(operations))
            
            return {
                'data': result,
                'stats': stats
            }
        return wrapper
    return decorator

def analyze(data: Any, operations: Set[str]) -> Dict[str, Dict[str, float]]:
    """
    分析数据中的数值型叶节点，返回统计结果
    
    参数:
        data: 要分析的数据
        operations: 要执行的统计操作集合
    
    返回:
        统计结果字典
    """
    # 收集所有数值型数据
    numeric_values = defaultdict(list)
    
    def collect_numeric_values(data, path="root"):
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            numeric_values[path].append(data)
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                collect_numeric_values(item, f"{path}[{i}]")
        elif isinstance(data, dict):
            for key, value in data.items():
                collect_numeric_values(value, f"{path}.{key}" if path != "root" else key)
    
    collect_numeric_values(data)
    
    # 计算统计特征
    stats_results = {}
    
    for path, values in numeric_values.items():
        if not values:
            continue
            
        path_stats = {}
        
        # 求和
        if 'SUM' in operations:
            path_stats['SUM'] = sum(values)
        
        # 均值
        if 'AVG' in operations:
            path_stats['AVG'] = sum(values) / len(values)
        
        # 方差
        if 'VAR' in operations:
            mean = sum(values) / len(values)
            path_stats['VAR'] = sum((x - mean) ** 2 for x in values) / len(values)
        
        # 均方根误差 (相对于0的RMSE)
        if 'RMSE' in operations:
            path_stats['RMSE'] = math.sqrt(sum(x ** 2 for x in values) / len(values))
        
        stats_results[path] = path_stats
    
    return stats_results

# 示例：如何使用装饰器
if __name__ == "__main__":
    from homework2 import DataSampler
    
    # 创建数据采样器实例
    sampler = DataSampler()
    
    # 使用装饰器装饰generate_sample方法
    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_data(schema, num_samples=1):
        return sampler.generate_samples(schema, num_samples)
    
    # 定义数据模式
    user_schema = {
        "id": "int",
        "score": "float",
        "is_active": "bool",
        "nested": {
            "value1": "int",
            "value2": "float"
        },
        "scores": {
            "type": "list",
            "items": "float",
            "length": 5
        }
    }
    
    # 生成数据并获取统计结果
    result = generate_data(user_schema, num_samples=10)
    
    # 打印原始数据
    print("原始数据:")
    print(result['data'])
    
    # 打印统计结果
    print("\n统计结果:")
    for path, stats in result['stats'].items():
        print(f"{path}:")
        for op, value in stats.items():
            print(f"  {op}: {value}")


