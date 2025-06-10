import random
import string
from datetime import datetime, timedelta
from typing import Any, List, Dict, Tuple, Union

class DataSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate_sample(self) -> Any:
        """根据kwargs生成一个随机样本"""
        return self._generate(self.kwargs)

    def generate_samples(self, num_samples: int) -> List[Any]:
        """生成指定数量的随机样本"""
        return [self.generate_sample() for _ in range(num_samples)]

    def _generate(self, schema: Dict) -> Any:
        """递归生成数据"""
        if isinstance(schema, dict):
            if 'type' in schema:
                return self._generate_by_type(schema)
            else:
                return {key: self._generate(value) for key, value in schema.items()}
        elif isinstance(schema, list):
            return [self._generate(item) for item in schema]
        elif isinstance(schema, tuple):
            return tuple(self._generate(item) for item in schema)
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

    def _generate_by_type(self, schema: Dict) -> Any:
        """根据类型生成数据"""
        data_type = schema['type']
        if data_type == 'int':
            return random.randint(schema.get('min', 0), schema.get('max', 100))
        elif data_type == 'float':
            return random.uniform(schema.get('min', 0.0), schema.get('max', 100.0))
        elif data_type == 'str':
            length = schema.get('length', 10)
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        elif data_type == 'bool':
            return random.choice([True, False])
        elif data_type == 'date':
            start_date = schema.get('start_date', datetime(2020, 1, 1))
            end_date = schema.get('end_date', datetime(2025, 1, 1))
            time_between_dates = end_date - start_date
            days_between_dates = time_between_dates.days
            random_number_of_days = random.randrange(days_between_dates)
            return start_date + timedelta(days=random_number_of_days)
        elif data_type == 'list':
            return self._generate(schema.get('items', []))
        elif data_type == 'tuple':
            return self._generate(schema.get('items', ()))
        elif data_type == 'dict':
            return self._generate(schema.get('items', {}))
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

if __name__ == "__main__":
    schema = {
        'type': 'dict',
        'items': {
            'id': {'type': 'int', 'min': 1, 'max': 1000},
            'name': {'type': 'str', 'length': 10},
            'age': {'type': 'int', 'min': 18, 'max': 65},
            'is_active': {'type': 'bool'},
            'created_at': {'type': 'date'},
            'address': {
                'type': 'dict',
                'items': {
                    'street': {'type': 'str', 'length': 20},
                    'city': {'type': 'str', 'length': 15},
                    'zip_code': {'type': 'str', 'length': 5}
                }
            },
            'phone_numbers': {
                'type': 'list',
                'items': {'type': 'str', 'length': 10}
            },
            'coordinates': {
                'type': 'tuple',
                'items': [
                    {'type': 'float', 'min': -180.0, 'max': 180.0},
                    {'type': 'float', 'min': -90.0, 'max': 90.0}
                ]
            }
        }
    }

    sampler = DataSampler(**schema)
    samples = sampler.generate_samples(5)
    for sample in samples:
        print(sample)
