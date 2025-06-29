
import math
import functools
from typing import Any, Dict, List, Tuple, Union, Callable, Set
from collections import defaultdict

def stats_decorator(*stat_operations):
    """
    带参数的装饰器，用于统计数据中数值型数据的特征
    
    参数:
        stat_operations: 统计操作的集合，可以是 'SUM', 'AVG', 'VAR', 'RMSE' 中的任意组合
    
    返回:
        装饰后的函数
    """
    # 验证操作参数
    valid_operations = {'SUM', 'AVG', 'VAR', 'RMSE'}
    for operation in stat_operations:
        if operation not in valid_operations:
            raise ValueError(f"不支持的操作: {operation}，支持的操作有: {valid_operations}")
    
    def decorator(target_func):
        @functools.wraps(target_func)
        def wrapper(*args, **kwargs):
            # 执行原函数获取数据
            func_result = target_func(*args, **kwargs)
            
            # 分析数据
            stat_result = analyze_numeric_stats(func_result, set(stat_operations))
            
            return {
                'data': func_result,
                'stats': stat_result
            }
        return wrapper
    return decorator

def analyze_numeric_stats(data: Any, stat_operations: Set[str]) -> Dict[str, Dict[str, float]]:
    """
    分析数据中的数值型叶节点，返回统计结果
    
    参数:
        data: 要分析的数据
        stat_operations: 要执行的统计操作集合
    
    返回:
        统计结果字典
    """
    # 收集所有数值型数据
    numeric_leaf_values = defaultdict(list)
    
    def collect_numeric_leaves(current_data, current_path="root"):
        if isinstance(current_data, (int, float)) and not isinstance(current_data, bool):
            numeric_leaf_values[current_path].append(current_data)
        elif isinstance(current_data, (list, tuple)):
            for idx, item in enumerate(current_data):
                collect_numeric_leaves(item, f"{current_path}[{idx}]")
        elif isinstance(current_data, dict):
            for key, value in current_data.items():
                collect_numeric_leaves(value, f"{current_path}.{key}" if current_path != "root" else key)
    
    collect_numeric_leaves(data)
    
    # 计算统计特征
    stats_results = {}
    
    for path, values in numeric_leaf_values.items():
        if not values:
            continue
                
        path_stats = {}
        
        # 求和
        if 'SUM' in stat_operations:
            path_stats['SUM'] = sum(values)
        
        # 均值
        if 'AVG' in stat_operations:
            path_stats['AVG'] = sum(values) / len(values)
        
        # 方差
        if 'VAR' in stat_operations:
            mean_value = sum(values) / len(values)
            path_stats['VAR'] = sum((x - mean_value) ** 2 for x in values) / len(values)
        
        # 均方根误差 (相对于0的RMSE)
        if 'RMSE' in stat_operations:
            path_stats['RMSE'] = math.sqrt(sum(x ** 2 for x in values) / len(values))
        
        stats_results[path] = path_stats
    
    return stats_results

# 示例：如何使用装饰器
if __name__ == "__main__":
    from homework2 import DataSampler
    
    # 创建数据采样器实例
    data_sampler = DataSampler()
    
    # 使用装饰器装饰generate_sample方法
    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_data_by_schema(data_schema, sample_count=1):
        return data_sampler.generate_samples(data_schema, sample_count)
    
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
    result = generate_data_by_schema(user_schema, sample_count=10)
    
    # 打印原始数据
    print("原始数据:")
    print(result['data'])
    
    # 打印统计结果
    print("\n统计结果:")
    for path, stats in result['stats'].items():
        print(f"{path}:")
        for op, value in stats.items():
            print(f"  {op}: {value}")


