import math
import numpy as np
from typing import Dict, List, Any, Callable, Set, Union, Tuple
from assignment2 import DataSampler

def stats_decorator(*operations: str):
    """
    带参数的修饰器，用于统计数据集中数值型数据的统计特征
    
    支持的统计操作:
    - SUM: 求和
    - AVG: 均值
    - VAR: 方差
    - RMSE: 均方根误差
    
    Args:
        *operations: 可选的统计操作，可以是"SUM", "AVG", "VAR", "RMSE"中的任意组合
        
    Returns:
        装饰器函数
    """
    # 检查传入的操作是否有效
    valid_operations = {"SUM", "AVG", "VAR", "RMSE"}
    for op in operations:
        if op not in valid_operations:
            raise ValueError(f"不支持的统计操作: {op}。支持的操作有: {valid_operations}")
    
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # 调用原始函数获取数据样本
            data_samples = func(*args, **kwargs)
            
            # 分析数据并计算统计结果
            stats_results = analyze(data_samples, set(operations))
            
            # 返回一个包含原始数据和统计结果的字典
            return {
                "data": data_samples,
                "stats": stats_results
            }
        return wrapper
    return decorator

def analyze(data: Any, operations: Set[str]) -> Dict[str, Any]:
    """
    分析数据中的数值型叶节点，返回统计结果
    
    Args:
        data: 要分析的数据
        operations: 要执行的统计操作集合
        
    Returns:
        包含统计结果的字典
    """
    # 提取所有数值型叶节点
    numeric_values = extract_numeric_values(data)
    
    if not numeric_values:
        return {"message": "未找到数值型数据"}
    
    # 计算统计结果
    results = {}
    
    if "SUM" in operations:
        results["SUM"] = sum(numeric_values)
        
    if "AVG" in operations:
        results["AVG"] = sum(numeric_values) / len(numeric_values)
        
    if "VAR" in operations:
        if len(numeric_values) > 1:
            mean = sum(numeric_values) / len(numeric_values)
            variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
            results["VAR"] = variance
        else:
            results["VAR"] = 0
        
    if "RMSE" in operations:
        # RMSE通常是预测值与实际值之间的差异
        # 在这里，我们将计算数值与其均值之间的均方根误差
        if len(numeric_values) > 1:
            mean = sum(numeric_values) / len(numeric_values)
            rmse = math.sqrt(sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values))
            results["RMSE"] = rmse
        else:
            results["RMSE"] = 0
    
    return results

def extract_numeric_values(data: Any) -> List[Union[int, float]]:
    """
    递归提取数据结构中的所有数值型叶节点
    
    Args:
        data: 要分析的数据
        
    Returns:
        包含所有数值型叶节点的列表
    """
    numeric_values = []
    
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        # 找到数值型叶节点
        numeric_values.append(data)
    elif isinstance(data, (list, tuple)):
        # 处理列表和元组
        for item in data:
            numeric_values.extend(extract_numeric_values(item))
    elif isinstance(data, dict):
        # 处理字典
        for value in data.values():
            numeric_values.extend(extract_numeric_values(value))
    
    return numeric_values

# 使用修饰器装饰DataSampler.generate_samples方法
@stats_decorator("SUM", "AVG", "VAR", "RMSE")
def generate_samples_with_stats(sampler: DataSampler, schema: Dict[str, Any], count: int = 1) -> List[Dict[str, Any]]:
    """
    生成数据样本并计算统计特征
    
    Args:
        sampler: DataSampler实例
        schema: 数据结构模式
        count: 样本数量
        
    Returns:
        生成的数据样本列表
    """
    return sampler.generate_samples(schema, count)

# 示例用法
def main():
    # 实例化数据生成器
    sampler = DataSampler()
    
    # 定义数据结构模式
    user_schema = {
        "id": {"type": "int", "min": 1000, "max": 9999},
        "age": {"type": "int", "min": 18, "max": 80},
        "height": {"type": "float", "min": 150.0, "max": 200.0, "precision": 1},
        "weight": {"type": "float", "min": 40.0, "max": 120.0, "precision": 1},
        "is_active": {"type": "bool"},
        "scores": {"type": "list", "item_type": {"type": "int", "min": 0, "max": 100}, "length": 5},
        "coordinates": {"type": "tuple", "item_type": {"type": "float", "min": -90, "max": 90}, "length": 2}
    }
    
    # 使用不同的统计组合生成并分析数据
    print("1. 生成数据并计算所有统计特征 (SUM, AVG, VAR, RMSE):")
    @stats_decorator("SUM", "AVG", "VAR", "RMSE")
    def generate_all_stats(sampler, schema, count):
        return sampler.generate_samples(schema, count)
    
    result_all = generate_all_stats(sampler, user_schema, 5)
    print("生成的样本数量:", len(result_all["data"]))
    print("统计结果:", result_all["stats"])
    print()
    
    print("2. 只计算均值和方差:")
    @stats_decorator("AVG", "VAR")
    def generate_avg_var(sampler, schema, count):
        return sampler.generate_samples(schema, count)
    
    result_avg_var = generate_avg_var(sampler, user_schema, 5)
    print("统计结果:", result_avg_var["stats"])
    print()
    
    print("3. 只计算求和:")
    @stats_decorator("SUM")
    def generate_sum(sampler, schema, count):
        return sampler.generate_samples(schema, count)
    
    result_sum = generate_sum(sampler, user_schema, 5)
    print("统计结果:", result_sum["stats"])
    print()
    
    print("4. 直接使用generate_samples_with_stats函数:")
    result_direct = generate_samples_with_stats(sampler, user_schema, 5)
    print("统计结果:", result_direct["stats"])

if __name__ == "__main__":
    main() 