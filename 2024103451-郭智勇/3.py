import math
import functools
import numpy as np
import random
import string
from copy import deepcopy

def generate_random_samples(**kwargs):
    """
    生成随机嵌套数据的样本集合。
    参数:
        structure: 指定嵌套数据结构的配置
        count: 需要生成的样本数量
        max_nesting_depth: 最大嵌套深度（可选，默认为3）
        rng: 随机数生成器（可选，默认使用random）

    返回:
        list: 随机生成的嵌套数据样本集合
    """
    structure = kwargs.get('structure', [{'type': 'list', 'length': 10}])
    count = kwargs.get('count', 1)
    max_depth = kwargs.get('max_nesting_depth', 3)
    rng = kwargs.get('rng', random)

    samples = []

    for _ in range(count):
        sample = generate_random_data(structure, max_depth, rng)
        samples.append(sample)

    return samples

def generate_random_data(structure, max_depth, rng, current_depth=0):
    data = []

    for layer in structure:
        if current_depth >= max_depth:
            data.append(generate_type(layer, rng))
        else:
            data_type = layer.get('type', 'list')
            if data_type == 'list':
                length = layer.get('length', rng.randint(1, 10))
                nested_structure = layer.get('elements', [{'type': 'int'}])
                data.append([generate_random_data(nested_structure, max_depth, rng, current_depth + 1) 
                             for _ in range(length)])
            elif data_type == 'dict':
                keys = layer.get('keys', rng.randint(1, 5))
                key_type = layer.get('key_type', 'str')
                nested_structure = layer.get('values', [{'type': 'int'}])
                data_dict = {}
                for _ in range(keys):
                    key = generate_type({'type': key_type}, rng)
                    value = generate_random_data(nested_structure, max_depth, rng, current_depth + 1)
                    data_dict[key] = value
                data.append(data_dict)
            else:
                data.append(generate_type(layer, rng))

    return data[0] if len(data) == 1 else data

def generate_type(config, rng):
    data_type = config.get('type', 'int')

    if data_type == 'int':
        return rng.randint(config.get('min', 0), config.get('max', 100))
    elif data_type == 'float':
        return rng.uniform(config.get('min', 0.0), config.get('max', 100.0))
    elif data_type == 'str':
        length = config.get('length', rng.randint(1, 10))
        letters = string.ascii_letters
        return ''.join(rng.choice(letters) for _ in range(length))
    elif data_type == 'bool':
        return rng.choice([True, False])
    elif data_type == 'none':
        return None
    elif data_type == 'custom':
        return config.get('generator', lambda: None)()

    return rng.randint(0, 100)

# 带参数的函数修饰器，用于统计数值型数据
def data_statistics(*stats):
    """
    带参数的修饰器，用于对生成样本中的数值型数据进行统计
    参数:
        stats: 允许的统计项组合，可以是 'SUM', 'AVG', 'VAR', 'RMSE'

    返回:
        装饰器函数
    """
    # 检查统计项参数的有效性
    valid_stats = ['SUM', 'AVG', 'VAR', 'RMSE']
    for stat in stats:
        if stat not in valid_stats:
            raise ValueError(f"无效的统计项：{stat}。有效的统计项包括：{valid_stats}")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原始函数获取样本数据
            samples = func(*args, **kwargs)
            # 统计所有数值型数据
            numeric_values = extract_numeric_values(samples)
            # 计算统计项
            results = {}
            if 'SUM' in stats:
                results['SUM'] = sum(numeric_values)
            if 'AVG' in stats:
                results['AVG'] = sum(numeric_values) / len(numeric_values) if numeric_values else 0
            if 'VAR' in stats:
                if len(numeric_values) < 2:
                    results['VAR'] = 0  # 方差需要至少两个数据点
                else:
                    mean = results.get('AVG', sum(numeric_values) / len(numeric_values))
                    results['VAR'] = sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1)
            if 'RMSE' in stats:
                if 'AVG' not in results:
                    mean = sum(numeric_values) / len(numeric_values) if numeric_values else 0
                else:
                    mean = results['AVG']
                results['RMSE'] = math.sqrt(sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)) if numeric_values else 0
            return results
        return wrapper
    return decorator

def extract_numeric_values(data):
    """
    从嵌套数据结构中提取所有数值型数据
    参数:
        data: 嵌套数据结构

    返回:
        list: 提取的数值型数据列表
    """
    numeric_values = []

    def _extract(data):
        if isinstance(data, (int, float)):
            numeric_values.append(data)
        elif isinstance(data, (list, tuple)):
            for item in data:
                _extract(item)
        elif isinstance(data, dict):
            for value in data.values():
                _extract(value)

    _extract(data)
    return numeric_values

# 示例用法
@data_statistics('SUM', 'AVG', 'VAR', 'RMSE')
def generate_and_statistics(**kwargs):
    return generate_random_samples(**kwargs)

# 调用示例
if __name__ == "__main__":
    structure_config = [{'type': 'list', 'length': 5, 'elements': [{'type': 'int'}, {'type': 'float'}]}]
    stats_results = generate_and_statistics(structure=structure_config, count=5)
    print(stats_results)