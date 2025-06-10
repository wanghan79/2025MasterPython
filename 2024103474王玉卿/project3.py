import random
import string
import math
from functools import wraps

def random_value(type_spec):
    if type_spec == 'int':
        return random.randint(0, 100)
    elif type_spec == 'float':
        return round(random.uniform(0, 100), 2)
    elif type_spec == 'str':
        return ''.join(random.choices(string.ascii_letters, k=5))
    elif type_spec == 'bool':
        return random.choice([True, False])
    elif isinstance(type_spec, dict):
        return {k: random_value(v) for k, v in type_spec.items()}
    elif isinstance(type_spec, list):
        value_type, length = type_spec
        return [random_value(value_type) for _ in range(length)]
    elif isinstance(type_spec, tuple):
        return tuple(random_value(t) for t in type_spec)
    else:
        raise ValueError(f"Unsupported type spec: {type_spec}")

#数值提取工具
def extract_numerical_values(data):
    results = []

    if isinstance(data, (int, float)):
        results.append(data)
    elif isinstance(data, dict):
        for value in data.values():
            results.extend(extract_numerical_values(value))
    elif isinstance(data, (list, tuple)):
        for item in data:
            results.extend(extract_numerical_values(item))
    return results

# 带参数的装饰器
def statistics_decorator(*stats):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            values = extract_numerical_values(samples)
            n = len(values)
            result = {}

            if not values:
                return samples, {s: None for s in stats}

            if 'sum' in stats:
                result['sum'] = sum(values)
            if 'avg' in stats:
                result['avg'] = sum(values) / n
            if 'var' in stats:
                mean = sum(values) / n
                result['var'] = sum((x - mean) ** 2 for x in values) / n
            if 'rmse' in stats:
                mean = sum(values) / n
                result['rmse'] = math.sqrt(sum((x - mean) ** 2 for x in values) / n)

            return samples, result
        return wrapper
    return decorator

# 使用装饰器修饰函数
@statistics_decorator('sum', 'avg', 'var', 'rmse')
def generate_samples(sample_type: dict, **kwargs):
    sample_count = kwargs.get('sample_count', 1)
    return [random_value(sample_type) for _ in range(sample_count)]


if __name__ == "__main__":
    schema = {
        'user_id': 'int',
        'score': 'float',
        'status': 'bool',
        'tags': ['str', 2],
        'nested': {
            'x': 'int',
            'y': 'float'
        }
    }

    data, stats = generate_samples(schema, sample_count=5)

    print("=== 样本数据 ===")
    for i, d in enumerate(data):
        print(f"{i+1}: {d}")

    print("\n=== 数值统计结果 ===")
    for k, v in stats.items():
        print(f"{k.upper()}: {v:.4f}")
