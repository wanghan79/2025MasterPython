import math
import pprint
import random
from functools import wraps
from collections import defaultdict


def stats_calculator(*metrics):
    """
    带参数的装饰器，指定要计算的统计指标
    :param metrics: 可变参数，指定统计指标（默认全部计算）
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)

            # 默认计算所有支持的指标
            supported_metrics = {
                'sum': lambda vals: sum(vals),
                'mean': lambda vals: sum(vals) / len(vals) if vals else 0,
                'variance': lambda vals: sum((x - (sum(vals) / len(vals))) ** 2 for x in vals) / len(vals) if len(
                    vals) > 1 else 0,
                'rmse': lambda vals: math.sqrt(sum(x ** 2 for x in vals) / len(vals)) if vals else 0
            }

            # 如果未指定指标，则计算全部
            selected_metrics = supported_metrics if not metrics else {
                m: supported_metrics[m] for m in metrics if m in supported_metrics
            }

            # 收集所有相同字段的值（忽略城镇编号）
            field_values = defaultdict(list)

            for sample in samples:
                # 每个样本是一个字典，键为 "town0", "town1" 等
                for town_key, town_data in sample.items():
                    # 遍历展平后的所有字段
                    for field_path, value in _flatten_dict(town_data):
                        if isinstance(value, (int, float)):
                            # 提取字段名（忽略城镇编号）
                            field_name = field_path.split('.', 1)[1] if '.' in field_path else field_path
                            field_values[field_name].append(value)

            # 计算选择的指标
            stats = {}
            for field_name, values in field_values.items():
                stats[field_name] = {
                    m: calc(values) for m, calc in selected_metrics.items()
                }

            return {'samples': samples, 'stats': stats}

        return wrapper

    return decorator


def _flatten_dict(d, parent_key='', sep='.'):
    """递归展平嵌套字典，生成 (路径, 值) 对"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep))
        else:
            items.append((new_key, v))
    return items


def DataGenerate(num, **kwargs):
    result = []
    for i in range(num):
        element = {}
        for key, value in kwargs.items():
            # 处理值（如果是字典则递归处理）
            processed_value = _process_value(value) if isinstance(value, dict) else value
            element[f"{key}{i}"] = processed_value
        result.append(element)
    return result


def _process_value(data):
    if isinstance(data, dict):
        return {k: _process_value(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        if all(isinstance(x, int) for x in data):
            return random.randint(data[0], data[1])
        elif all(isinstance(x, float) for x in data):
            return random.uniform(data[0], data[1])
    return data


@stats_calculator('sum', 'mean', 'variance', 'rmse')
def generate_town_data(num, **kwargs):
    return DataGenerate(num, **kwargs)


# 测试
if __name__ == "__main__":
    data_format = {
        'town': {
            'school': {
                'teachers': (50, 70),
                'students': (800, 1200),
                'others': (20, 40),
                'money': (410000.5, 986553.1)
            },
            'hospital': {
                'doctors': (40, 60),  # 修正拼写错误
                'nurses': (60, 80),
                'patients': (200, 300),
                'money': (110050.5, 426553.4)
            },
            'supermarket': {
                'sailers': (80, 150),
                'shop': (30, 60),
                'money': (310000.3, 7965453.4)
            }
        }
    }

    result = generate_town_data(5, **data_format)

    print("=" * 50)
    print("样本数据示例:")
    print("=" * 50)
    pprint.pprint(result['samples'][0])  # 打印第一个样本

    print("\n" + "=" * 50)
    print("统计结果:")
    print("=" * 50)
    pprint.pprint(result['stats'])