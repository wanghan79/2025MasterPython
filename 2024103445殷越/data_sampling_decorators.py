import random
import string
import numpy as np
from functools import wraps
from typing import Callable, Any, Dict, List, Union, Literal
from data_sampling import create_random_object

def sample_generator(num_samples: int = 1, **kwargs):
    """
    生成指定数量的随机样本
    
    Args:
        num_samples: 要生成的样本数量
        **kwargs: 传递给create_random_object的参数
    
    Returns:
        生成的样本列表
    """
    return [create_random_object(**kwargs) for _ in range(num_samples)]

def extract_field_values(obj: Any, field_path: str = '') -> Dict[str, List[float]]:
    """
    从对象中提取字段值
    
    Args:
        obj: 输入对象，可以是数值、字典或列表
        field_path: 当前字段的路径
    
    Returns:
        字段路径到值的映射字典
    """
    result = {}
    
    if isinstance(obj, (int, float)):
        if field_path:
            result[field_path] = [float(obj)]
        else:
            result['value'] = [float(obj)]
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{field_path}.{key}" if field_path else key
            result.update(extract_field_values(value, new_path))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            new_path = f"{field_path}[{i}]" if field_path else f"[{i}]" 
            result.update(extract_field_values(item, new_path))
    
    return result

def statistical_decorator(stat_types: Union[str, List[str]] = 'mean'):
    """
    统一的统计装饰器，支持不同统计方式的任意组合
    
    Args:
        stat_types: 统计类型，可以是单个字符串或字符串列表
                   可选值：'mean', 'variance', 'rmse', 'sum'
    """
    # 如果传入的是单个字符串，转换为列表
    if isinstance(stat_types, str):
        stat_types = [stat_types]
    
    # 验证所有统计类型都是支持的
    supported_types = {'mean', 'variance', 'rmse', 'sum'}
    for stat_type in stat_types:
        if stat_type not in supported_types:
            raise ValueError(f"Unsupported statistical type: {stat_type}")
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            if not isinstance(results, (list, tuple)):
                results = [results]
            
            # 提取所有字段的值
            field_values = {}
            for result in results:
                for field, values in extract_field_values(result).items():
                    if field not in field_values:
                        field_values[field] = []
                    field_values[field].extend(values)
            
            # 对每个统计类型和每个字段进行统计
            stats = {}
            for stat_type in stat_types:
                stats[stat_type] = {}
                for field, values in field_values.items():
                    if not values:
                        stats[stat_type][field] = None
                        continue
                        
                    if stat_type == 'mean':
                        stats[stat_type][field] = float(np.mean(values))
                    elif stat_type == 'variance':
                        stats[stat_type][field] = float(np.var(values))
                    elif stat_type == 'rmse':
                        stats[stat_type][field] = float(np.sqrt(np.mean(np.square(values))))
                    elif stat_type == 'sum':
                        stats[stat_type][field] = float(np.sum(values))
            
            # 如果只有一种统计类型，直接返回该类型的结果
            if len(stat_types) == 1:
                return stats[stat_types[0]]
            
            return stats
        return wrapper
    return decorator

# 示例使用
if __name__ == "__main__":
    # 示例1：单一统计类型
    @statistical_decorator(stat_types='mean')
    def generate_random_numbers_mean(num_samples: int = 5):
        return sample_generator(num_samples, value=int)
    
    # 示例2：多种统计类型组合
    @statistical_decorator(stat_types=['mean', 'variance', 'sum'])
    def generate_random_numbers_multi(num_samples: int = 5):
        return sample_generator(num_samples, value=int)
    
    # 示例3：所有统计类型
    @statistical_decorator(stat_types=['mean', 'variance', 'rmse', 'sum'])
    def generate_random_numbers_all(num_samples: int = 5):
        return sample_generator(num_samples, value=int)
    
    # 示例4：复杂对象的多种统计
    @statistical_decorator(stat_types=['mean', 'variance'])
    def generate_complex_objects(num_samples: int = 5):
        return sample_generator(
            num_samples,
            age=int,
            score=float,
            name=str,
            nested=dict(
                nested_age=int,
                nested_name=str
            )
        )
    
    # 测试代码
    print("Single stat (mean):", generate_random_numbers_mean(5))
    print("\nMultiple stats (mean, variance, sum):")
    result_multi = generate_random_numbers_multi(5)
    for stat_type, values in result_multi.items():
        print(f"  {stat_type}: {values}")
    
    print("\nAll stats:")
    result_all = generate_random_numbers_all(5)
    for stat_type, values in result_all.items():
        print(f"  {stat_type}: {values}")
    
    print("\nComplex objects (mean, variance):")
    result_complex = generate_complex_objects(3)
    for stat_type, values in result_complex.items():
        print(f"  {stat_type}: {values}") 