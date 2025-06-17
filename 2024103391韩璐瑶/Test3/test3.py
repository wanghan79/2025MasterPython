import random
import string
import math
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, List


# ===== 带参数修饰器 =====
def stats_decorator(*metrics):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            values = []

            # 递归提取所有数值型数据（int 和 float）
            def extract_numbers(d):
                if isinstance(d, dict):
                    for v in d.values():
                        extract_numbers(v)
                elif isinstance(d, (list, tuple)):
                    for item in d:
                        extract_numbers(item)
                elif isinstance(d, (int, float)):
                    values.append(d)

            for item in data:
                extract_numbers(item)

            result = {}
            if 'SUM' in metrics:
                result['SUM'] = sum(values)
            if 'AVG' in metrics:
                result['AVG'] = sum(values) / len(values) if values else 0
            if 'VAR' in metrics:
                mean = sum(values) / len(values) if values else 0
                result['VAR'] = sum((x - mean) ** 2 for x in values) / len(values) if values else 0
            if 'RMSE' in metrics:
                result['RMSE'] = math.sqrt(sum(x ** 2 for x in values) / len(values)) if values else 0

            print(f"\n统计结果 ({', '.join(metrics)}):")
            for k, v in result.items():
                print(f"  {k}: {v:.4f}")
            return data
        return wrapper
    return decorator


# ===== 数据生成器类 =====
class DataSampler:
    def __init__(self):
        pass

    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')  # 可根据需要修改统计项
    def sample(self, structure: Any, num: int = 1) -> List[Any]:
        return [self._generate(structure) for _ in range(num)]

    def _generate(self, descriptor: Any) -> Any:
        if isinstance(descriptor, dict):
            return {key: self._generate(value) for key, value in descriptor.items()}
        elif isinstance(descriptor, list):
            return [self._generate(item) for item in descriptor]
        elif isinstance(descriptor, tuple):
            return tuple(self._generate(item) for item in descriptor)
        elif isinstance(descriptor, str):
            if descriptor.startswith('int:'):
                low, high = map(int, descriptor.split(':')[1].split(','))
                return random.randint(low, high)
            elif descriptor.startswith('float:'):
                low, high = map(float, descriptor.split(':')[1].split(','))
                return round(random.uniform(low, high), 2)
            elif descriptor.startswith('str:'):
                length = int(descriptor.split(':')[1])
                return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            elif descriptor == 'bool':
                return random.choice([True, False])
            elif descriptor.startswith('date:'):
                start_str, end_str = descriptor.split(':')[1].split(',')
                start_date = datetime.strptime(start_str.strip(), '%Y-%m-%d')
                end_date = datetime.strptime(end_str.strip(), '%Y-%m-%d')
                delta = (end_date - start_date).days
                random_days = random.randint(0, delta)
                return (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
            else:
                return descriptor
        else:
            return descriptor


# ===== 测试运行部分 =====
if __name__ == '__main__':
    structure = {
        "user_id": "int:1000,9999",
        "name": "str:8",
        "active": "bool",
        "signup_date": "date:2022-01-01,2023-12-31",
        "profile": {
            "age": "int:18,60",
            "height": "float:150,200",
            "scores": ["float:0,1", "float:0,1"]
        },
        "tags": ("str:4", "str:4")
    }

    sampler = DataSampler()
    samples = sampler.sample(structure, num=5)

    from pprint import pprint
    pprint(samples)
