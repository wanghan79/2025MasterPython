import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union

class DataSampler:
    def __init__(self):
        pass

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
                return descriptor  # literal value
        else:
            return descriptor  # literal value

# 示例结构定义
structure = {
    "user_id": "int:1000,9999",
    "name": "str:8",
    "active": "bool",
    "signup_date": "date:2022-01-01,2023-12-31",
    "profile": {
        "age": "int:18,60",
        "height": "float:150,200",
        "hobbies": ["str:5", "str:6"]
    },
    "tags": ("str:4", "str:4")
}

# 生成样本
sampler = DataSampler()
samples = sampler.sample(structure, num=3)

# 输出
from pprint import pprint
pprint(samples)
