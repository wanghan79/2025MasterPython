TTP://ONECLICK.LENOVO.COM.CN

import math
import pprint
import random
from functools import wraps


def stats_calculator(*metrics):
    """
    带参数的装饰器，指定要计算的统计指标（如 'sum', 'mean', 'variance', 'rmse'）
    :param metrics: 可变参数，指定统计指标（默认全部计算）
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)

            # 默认计算所有支持的指标
            supported_metrics = {
                'sum': lambda vals: sum(vals),
                'mean': lambda vals: sum(vals) / len(vals),
                'variance': lambda vals: sum((x - (sum(vals) / len(vals))) ** 2 for x in vals) / len(vals),
                'rmse': lambda vals: math.sqrt(sum((x - (sum(vals) / len(vals))) ** 2 for x in vals) / len(vals))
            }

            # 如果未指定指标，则计算全部
            selected_metrics = supported_metrics if not metrics else {
                m: supported_metrics[m] for m in metrics if m in supported_metrics
            }

            # 遍历所有样本，提取所有数值字段（自动探测）
            all_values = {}
            for sample in samples:
                for town_data in sample.values():  # 遍历每个 townX
                    for field_path, value in _flatten_dict(town_data):
                        if isinstance(value, (int, float)):
                            all_values.setdefault(field_path, []).append(value)

            # 计算选择的指标
            stats = {}
            for field_path, values in all_values.items():
                stats[field_path] = {
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
@stats_calculator('sum', 'mean','variance','rmse')
def generate_town_data(num, **kwargs):
    return DataGenerate(num, **kwargs)
data_format = {'town':
                       {'school':
                            {'teachers':(50,70),
                             'students':(800,1200),
                             'others':(20,40),
                            'money':(410000.5,986553.1)},
                        'hospital':
                            {'docters':(40,60),
                             'nurses':(60,80),
                             'patients':(200,300),
                            'money':(110050.5,426553.4)},
                        'supermarket':
                            {'sailers':(80,150),
                             'shop':(30,60),
                            'money':(310000.3,7965453.4)}
                        }
                   }
result = generate_town_data(5, **data_format)
pprint.pprint(result['stats'])