import random
import string
import math
from datetime import datetime, timedelta
from functools import wraps


def stats_decorator(*stats_args):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            numeric_data = []

            # 递归收集所有数值型数据
            def collect_numbers(d):
                if isinstance(d, (int, float)):
                    numeric_data.append(d)
                elif isinstance(d, (list, tuple)):
                    for item in d:
                        collect_numbers(item)
                elif isinstance(d, dict):
                    for v in d.values():
                        collect_numbers(v)

            collect_numbers(data)

            results = {}
            if not numeric_data:
                return results

            if 'SUM' in stats_args:
                results['SUM'] = sum(numeric_data)
            if 'AVG' in stats_args:
                results['AVG'] = sum(numeric_data) / len(numeric_data)
            if 'VAR' in stats_args:
                mean = sum(numeric_data) / len(numeric_data)
                results['VAR'] = sum((x - mean) ** 2 for x in numeric_data) / len(numeric_data)
            if 'RMSE' in stats_args:
                mean = sum(numeric_data) / len(numeric_data)
                results['RMSE'] = math.sqrt(sum((x - mean) ** 2 for x in numeric_data) / len(numeric_data))

            return {'data': data, 'stats': results} if results else data

        return wrapper

    return decorator


# 使用装饰器的示例
@stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def DataSampler(**kwargs):
    dtype = kwargs.get('type')
    if dtype == 'int':
        return random.randint(*kwargs.get('range', (0, 100)))
    elif dtype == 'float':
        return random.uniform(*kwargs.get('range', (0, 100)))
    elif dtype == 'str':
        length = kwargs.get('length', 10)
        chars = kwargs.get('chars', string.ascii_letters + string.digits)
        return ''.join(random.choices(chars, k=length))
    elif dtype == 'bool':
        return random.choice([True, False])
    elif dtype == 'date':
        start = kwargs.get('start', datetime(2000, 1, 1))
        end = kwargs.get('end', datetime(2023, 1, 1))
        return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))
    elif dtype == 'list':
        return [DataSampler(**kwargs['item']) for _ in range(kwargs.get('size', 3))]
    elif dtype == 'tuple':
        return tuple(DataSampler(**kwargs['item']) for _ in range(kwargs.get('size', 3)))
    elif dtype == 'dict':
        return {k: DataSampler(**v) for k, v in kwargs['fields'].items()}
    return None


# 测试代码
user_data = {
    'type': 'dict',
    'fields': {
        'id': {'type': 'int', 'range': (1, 1000)},
        'name': {'type': 'str', 'length': 8},
        'scores': {'type': 'list', 'item': {'type': 'float', 'range': (0, 100)}, 'size': 3},
        'active': {'type': 'bool'},
    }
}

result = DataSampler(**user_data)
print("生成的数据:", result['data'])
print("统计结果:", result['stats'])