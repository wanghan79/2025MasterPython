import random
import string
import math
from functools import wraps
from homework2 import generate_nested_samples

def statistics_decorator(*stats_types):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            numeric_values = extract_numeric_values(samples)
            
            stats_result = {}
            if numeric_values:
                for stat_type in stats_types:
                    if stat_type == 'SUM':
                        stats_result['SUM'] = sum(numeric_values)
                    elif stat_type == 'AVG':
                        stats_result['AVG'] = sum(numeric_values) / len(numeric_values)
                    elif stat_type == 'VAR':
                        mean = sum(numeric_values) / len(numeric_values)
                        stats_result['VAR'] = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
                    elif stat_type == 'RMSE':
                        mean = sum(numeric_values) / len(numeric_values)
                        variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
                        stats_result['RMSE'] = math.sqrt(variance)
            
            return samples, stats_result
        return wrapper
    return decorator

def extract_numeric_values(data):
    numeric_values = []
    
    if isinstance(data, dict):
        for value in data.values():
            numeric_values.extend(extract_numeric_values(value))
    elif isinstance(data, (list, tuple)):
        for item in data:
            numeric_values.extend(extract_numeric_values(item))
    elif isinstance(data, (int, float)):
        numeric_values.append(data)
    
    return numeric_values

@statistics_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def generate_with_all_stats(**kwargs):
    return generate_nested_samples(**kwargs)

@statistics_decorator('SUM', 'AVG')
def generate_with_sum_avg(**kwargs):
    return generate_nested_samples(**kwargs)

@statistics_decorator('VAR', 'RMSE')
def generate_with_var_rmse(**kwargs):
    return generate_nested_samples(**kwargs)

@statistics_decorator('AVG')
def generate_with_avg_only(**kwargs):
    return generate_nested_samples(**kwargs)

def test_decorator():
    print("=" * 50)
    print("修饰器功能测试")
    print("=" * 50)
    
    # 全部统计项测试
    samples, stats = generate_with_all_stats(
        count=5,
        id='int',
        score='float',
        grades=['int'],
        user_info={'age': 'int', 'weight': 'float'}
    )
    
    print(f"\n生成样本数: {len(samples)}")
    print("所有统计项:")
    for name, value in stats.items():
        print(f"  {name}: {value:.3f}")
    
    # 部分统计项测试
    samples, stats = generate_with_sum_avg(
        count=3,
        temp='float',
        humidity='int'
    )
    
    print(f"\n样本数: {len(samples)}")
    print("SUM和AVG:")
    for name, value in stats.items():
        print(f"  {name}: {value:.3f}")
    
    print(f"\n样本示例: {samples[0]}")

if __name__ == "__main__":
    test_decorator()