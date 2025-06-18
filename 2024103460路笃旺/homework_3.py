import math
import pprint
import random
from functools import wraps
from typing import Dict, List, Union, Tuple, Any, Callable, TypeVar, Optional

# 定义类型别名
T = TypeVar('T')
MetricFunc = Callable[[List[float]], float]
StatsResult = Dict[str, Dict[str, float]]


def stats_calculator(*metrics: str) -> Callable:
    """带参数的装饰器，用于计算统计指标

    Args:
        *metrics: 要计算的统计指标名称（如 'sum', 'mean', 'variance', 'rmse'）

    Returns:
        装饰器函数
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            samples = func(*args, **kwargs)

            # 定义支持的统计指标
            supported_metrics: Dict[str, MetricFunc] = {
                'sum': lambda vals: sum(vals),
                'mean': lambda vals: sum(vals) / len(vals),
                'variance': lambda vals: sum((x - (sum(vals) / len(vals))) ** 2 for x in vals) / len(vals),
                'rmse': lambda vals: math.sqrt(sum((x - (sum(vals) / len(vals))) ** 2 for x in vals) / len(vals))
            }

            # 选择要计算的指标
            selected_metrics = supported_metrics if not metrics else {
                m: supported_metrics[m] for m in metrics if m in supported_metrics
            }

            # 收集所有数值字段
            all_values: Dict[str, List[float]] = {}
            for sample in samples:
                for town_data in sample.values():
                    for field_path, value in _flatten_dict(town_data):
                        if isinstance(value, (int, float)):
                            all_values.setdefault(field_path, []).append(float(value))

            # 计算统计指标
            stats: StatsResult = {}
            for field_path, values in all_values.items():
                stats[field_path] = {
                    metric: calc(values) for metric, calc in selected_metrics.items()
                }

            return {'samples': samples, 'stats': stats}

        return wrapper

    return decorator


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> List[Tuple[str, Any]]:
    """递归展平嵌套字典

    Args:
        d: 要展平的字典
        parent_key: 父键名
        sep: 分隔符

    Returns:
        展平后的(键路径, 值)对列表
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep))
        else:
            items.append((new_key, v))
    return items


def generate_random_value(value_range: Union[Tuple[int, int], Tuple[float, float]]) -> Union[int, float]:
    """根据范围生成随机值

    Args:
        value_range: 取值范围元组 (min, max)

    Returns:
        生成的随机值
    """
    if all(isinstance(x, int) for x in value_range):
        return random.randint(value_range[0], value_range[1])
    elif all(isinstance(x, float) for x in value_range):
        return random.uniform(value_range[0], value_range[1])
    return value_range


def process_nested_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """处理嵌套字典，生成随机值

    Args:
        data: 输入的数据字典

    Returns:
        处理后的字典
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = process_nested_dict(value)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            result[key] = generate_random_value(value)
        else:
            result[key] = value
    return result


def generate_data(num: int, **kwargs) -> List[Dict[str, Any]]:
    """生成指定数量的随机数据

    Args:
        num: 需要生成的数据数量
        **kwargs: 数据格式定义

    Returns:
        生成的数据列表
    """
    result = []
    for i in range(num):
        element = {}
        for key, value in kwargs.items():
            processed_value = process_nested_dict(value) if isinstance(value, dict) else value
            element[f"{key}{i}"] = processed_value
        result.append(element)
    return result


@stats_calculator('sum', 'mean', 'variance', 'rmse')
def generate_town_data(num: int, **kwargs) -> List[Dict[str, Any]]:
    """生成城镇数据并计算统计指标

    Args:
        num: 需要生成的数据数量
        **kwargs: 数据格式定义

    Returns:
        生成的数据列表
    """
    return generate_data(num, **kwargs)


def main():
    # 数据格式定义
    data_format = {
        'town': {
            'school': {
                'teachers': (50, 70),
                'students': (800, 1200),
                'others': (20, 40),
                'money': (410000.5, 986553.1)
            },
            'hospital': {
                'docters': (40, 60),
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

    # 生成数据并计算统计指标
    result = generate_town_data(5, **data_format)

    # 打印统计结果
    print("\n=== 统计结果 ===")
    pprint.pprint(result['stats'])


if __name__ == '__main__':
    main()
