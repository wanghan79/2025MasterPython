
import random
import math
from functools import wraps
from typing import Any, Callable, List, Dict, Union


# ----------- 样本生成核心函数 -----------

def generate_random_data(spec: dict) -> Any:
    """
    根据类型说明生成一个随机值
    """
    dtype = spec['type']
    if dtype == int:
        return random.randint(*spec['range'])
    elif dtype == float:
        return random.uniform(*spec['range'])
    elif dtype == str:
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=spec['length']))
    elif dtype == list:
        return [generate_random_data(spec['subtype']) for _ in range(spec['length'])]
    elif dtype == tuple:
        return tuple(generate_random_data(spec['subtype']) for _ in range(spec['length']))
    elif dtype == dict:
        return {k: generate_random_data(v) for k, v in spec['fields'].items()}
    else:
        return None


def build_samples(sample_num: int, structure: dict) -> List[dict]:
    """
    构建多个嵌套结构样本
    """
    return [generate_random_data(structure) for _ in range(sample_num)]


# ----------- 修饰器与数值统计 -----------

def extract_numeric_values(data: Any, filter_fn: Callable[[Any], bool] = None) -> List[float]:
    """
    递归提取嵌套结构中的数值（int、float）
    可选：传入过滤函数，如筛选字段名或只保留 float 类型
    """
    result = []

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (int, float)):
                if filter_fn is None or filter_fn((k, v)):
                    result.append(float(v))
            else:
                result.extend(extract_numeric_values(v, filter_fn))
    elif isinstance(data, (list, tuple)):
        for item in data:
            result.extend(extract_numeric_values(item, filter_fn))
    elif isinstance(data, (int, float)):
        if filter_fn is None or filter_fn(('', data)):
            result.append(float(data))

    return result


def stat_decorator(*metrics: str, filter_fn: Callable = None, detailed: bool = False):
    """
    修饰器支持统计嵌套结构中所有数值型字段：
    - metrics: 'SUM', 'AVG', 'VAR', 'RMSE'
    - filter_fn: 可选的值过滤器
    - detailed: 是否每个样本分别统计
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            final_result = {}

            if detailed:
                final_result['individual'] = []
                for idx, sample in enumerate(samples):
                    values = extract_numeric_values(sample, filter_fn)
                    stats = compute_metrics(values, metrics)
                    final_result['individual'].append({f"Sample_{idx}": stats})

            # 总体统计
            all_values = extract_numeric_values(samples, filter_fn)
            final_result['summary'] = compute_metrics(all_values, metrics)

            return final_result
        return wrapper
    return decorator


def compute_metrics(values: List[float], metrics: Union[str, List[str]]) -> Dict[str, float]:
    """
    给定一组数值，计算对应的统计项
    """
    result = {}
    if not values:
        return {m: None for m in metrics}

    if 'SUM' in metrics:
        result['SUM'] = sum(values)
    if 'AVG' in metrics:
        result['AVG'] = sum(values) / len(values)
    if 'VAR' in metrics:
        avg = sum(values) / len(values)
        result['VAR'] = sum((x - avg) ** 2 for x in values) / len(values)
    if 'RMSE' in metrics:
        result['RMSE'] = math.sqrt(sum(x ** 2 for x in values) / len(values))
    return result


# ----------- 示例结构定义 & 使用 -----------

# 顶层结构，必须包含 'type' 和 'fields'
sample_structure = {
    'type': dict,
    'fields': {
        'device': {
            'type': dict,
            'fields': {
                'id': {'type': str, 'length': 8},
                'status': {'type': int, 'range': (0, 1)},
                'sensors': {
                    'type': list,
                    'length': 3,
                    'subtype': {
                        'type': dict,
                        'fields': {
                            'temperature': {'type': float, 'range': (-10.0, 40.0)},
                            'humidity': {'type': float, 'range': (0.0, 100.0)},
                            'code': {'type': str, 'length': 4}
                        }
                    }
                }
            }
        },
        'timestamp': {'type': str, 'length': 10},
        'signal_strength': {'type': float, 'range': (0.0, 5.0)}
    }
}



# ----------- 示例函数 + 装饰器修饰 -----------

@stat_decorator('SUM', 'AVG', 'VAR', 'RMSE', detailed=True)
def generate_complex_samples():
    return build_samples(sample_num=5, structure=sample_structure)


# ----------- 主程序入口 -----------

if __name__ == '__main__':
    result = generate_complex_samples()
    print("=== 每个样本统计 ===")
    for item in result['individual']:
        print(item)
    print("\n=== 汇总统计结果 ===")
    for key, val in result['summary'].items():
        print(f"{key}: {val:.4f}")

