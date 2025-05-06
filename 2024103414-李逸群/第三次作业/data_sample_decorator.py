import random
import string
import numpy as np
import pandas as pd
import json
from functools import wraps
from typing import Callable, Any, Dict, List, Union, Literal, Optional, Tuple, TypeVar
from data_sampling import create_random_object

T = TypeVar('T')

def sample_generator(num_samples: int = 1, **kwargs) -> List[Dict[str, Any]]:
  
    return [create_random_object(**kwargs) for _ in range(num_samples)]

def extract_field_values(obj: Any, field_path: str = '') -> Dict[str, List[float]]:
    
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

def statistical_decorator(stat_type: Literal['mean', 'variance', 'rmse', 'sum', 'min', 'max', 'median', 'std', 'percentile'] = 'mean', 
                         percentile: Optional[int] = None,
                         return_format: Literal['dict', 'dataframe', 'json'] = 'dict'):
    
    def decorator(func: Callable[..., T]) -> Callable[..., Union[Dict[str, float], pd.DataFrame, str]]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取原始函数结果
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
            
            # 对每个字段进行统计
            stats = {}
            for field, values in field_values.items():
                if not values:
                    stats[field] = None
                    continue
                
                # 转换为numpy数组以便计算
                np_values = np.array(values, dtype=float)
                
                if stat_type == 'mean':
                    stats[field] = float(np.mean(np_values))
                elif stat_type == 'variance':
                    stats[field] = float(np.var(np_values))
                elif stat_type == 'rmse':
                    stats[field] = float(np.sqrt(np.mean(np.square(np_values))))
                elif stat_type == 'sum':
                    stats[field] = float(np.sum(np_values))
                elif stat_type == 'min':
                    stats[field] = float(np.min(np_values))
                elif stat_type == 'max':
                    stats[field] = float(np.max(np_values))
                elif stat_type == 'median':
                    stats[field] = float(np.median(np_values))
                elif stat_type == 'std':
                    stats[field] = float(np.std(np_values))
                elif stat_type == 'percentile':
                    if percentile is None or not (1 <= percentile <= 99):
                        raise ValueError("百分位数必须在1到99之间")
                    stats[field] = float(np.percentile(np_values, percentile))
                else:
                    raise ValueError(f"不支持的统计类型: {stat_type}")
            
            # 根据指定格式返回结果
            if return_format == 'dict':
                return stats
            elif return_format == 'dataframe':
                return pd.DataFrame(list(stats.items()), columns=['Field', 'Value'])
            elif return_format == 'json':
                return json.dumps(stats, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的返回格式: {return_format}")
                
        return wrapper
    return decorator

def multi_stat_decorator(stat_types: List[str], 
                        percentile: Optional[int] = None,
                        return_format: Literal['dict', 'dataframe', 'json'] = 'dict'):
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取原始函数结果
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
            
            # 对每个字段计算多种统计量
            multi_stats = {}
            for field, values in field_values.items():
                if not values:
                    multi_stats[field] = {stat: None for stat in stat_types}
                    continue
                
                # 转换为numpy数组以便计算
                np_values = np.array(values, dtype=float)
                field_stats = {}
                
                for stat in stat_types:
                    if stat == 'mean':
                        field_stats['mean'] = float(np.mean(np_values))
                    elif stat == 'variance':
                        field_stats['variance'] = float(np.var(np_values))
                    elif stat == 'rmse':
                        field_stats['rmse'] = float(np.sqrt(np.mean(np.square(np_values))))
                    elif stat == 'sum':
                        field_stats['sum'] = float(np.sum(np_values))
                    elif stat == 'min':
                        field_stats['min'] = float(np.min(np_values))
                    elif stat == 'max':
                        field_stats['max'] = float(np.max(np_values))
                    elif stat == 'median':
                        field_stats['median'] = float(np.median(np_values))
                    elif stat == 'std':
                        field_stats['std'] = float(np.std(np_values))
                    elif stat == 'percentile':
                        if percentile is None or not (1 <= percentile <= 99):
                            raise ValueError("百分位数必须在1到99之间")
                        field_stats[f'percentile_{percentile}'] = float(np.percentile(np_values, percentile))
                    else:
                        raise ValueError(f"不支持的统计类型: {stat}")
                
                multi_stats[field] = field_stats
            
            # 根据指定格式返回结果
            if return_format == 'dict':
                return multi_stats
            elif return_format == 'dataframe':
                # 将嵌套字典转换为DataFrame
                rows = []
                for field, stats in multi_stats.items():
                    for stat_name, value in stats.items():
                        rows.append({'Field': field, 'Statistic': stat_name, 'Value': value})
                return pd.DataFrame(rows)
            elif return_format == 'json':
                return json.dumps(multi_stats, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的返回格式: {return_format}")
                
        return wrapper
    return decorator

def save_samples(file_path: str, format: Literal['csv', 'json', 'pickle'] = 'csv'):
    """
    保存样本数据装饰器
    
    Args:
        file_path: 保存文件路径
        format: 文件格式，可选值：'csv', 'json', 'pickle'
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取原始函数结果
            results = func(*args, **kwargs)
            
            # 确保结果是列表
            if not isinstance(results, (list, tuple)):
                results = [results]
            
            # 根据指定格式保存结果
            if format == 'csv':
                df = pd.DataFrame(results)
                df.to_csv(file_path, index=False, encoding='utf-8')
            elif format == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            elif format == 'pickle':
                pd.DataFrame(results).to_pickle(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {format}")
            
            print(f"样本数据已保存至: {file_path}")
            return results
            
        return wrapper
    return decorator

# 示例使用
if __name__ == "__main__":
    # 示例1：生成随机整数并计算不同统计量
    @statistical_decorator(stat_type='mean')
    def generate_random_numbers_mean(num_samples: int = 5):
        return sample_generator(num_samples, value=int)
    
    @statistical_decorator(stat_type='variance')
    def generate_random_numbers_var(num_samples: int = 5):
        return sample_generator(num_samples, value=int)
    
    @statistical_decorator(stat_type='rmse')
    def generate_random_numbers_rmse(num_samples: int = 5):
        return sample_generator(num_samples, value=int)
    
    @statistical_decorator(stat_type='sum')
    def generate_random_numbers_sum(num_samples: int = 5):
        return sample_generator(num_samples, value=int)
    
    @statistical_decorator(stat_type='percentile', percentile=75, return_format='json')
    def generate_random_numbers_percentile(num_samples: int = 10):
        return sample_generator(num_samples, value=int)
    
    # 示例2：生成复杂对象并计算多种统计量
    @multi_stat_decorator(stat_types=['mean', 'min', 'max', 'std'], return_format='dataframe')
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
    
    # 示例3：保存生成的样本
    @save_samples(file_path='e:\\pythonProject\\samples.csv', format='csv')
    def generate_and_save_samples(num_samples: int = 10):
        return sample_generator(
            num_samples,
            id=int,
            name=str,
            score=(float, {'min': 0, 'max': 100}),
            is_active=bool
        )
    
    # 测试代码
    print("随机数均值:", generate_random_numbers_mean(5))
    print("随机数方差:", generate_random_numbers_var(5))
    print("随机数RMSE:", generate_random_numbers_rmse(5))
    print("随机数总和:", generate_random_numbers_sum(5))
    print("\n随机数75百分位数(JSON格式):")
    print(generate_random_numbers_percentile(10))
    print("\n复杂对象多种统计量(DataFrame格式):")
    print(generate_complex_objects(3))
    
    # 生成并保存样本
    samples = generate_and_save_samples(10)
    print("\n生成的样本示例:")
    for sample in samples[:3]:
        print(sample)