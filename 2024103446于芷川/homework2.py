import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union

class DataSampler:
    def __init__(self):
        self.type_generators = {
            'int': self._generate_int,
            'float': self._generate_float,
            'str': self._generate_str,
            'bool': self._generate_bool,
            'date': self._generate_date,
            'list': self._generate_list,
            'tuple': self._generate_tuple,
            'dict': self._generate_dict
        }

    def _generate_int(self, min_val: int = -100, max_val: int = 100) -> int:
        return random.randint(min_val, max_val)

    def _generate_float(self, min_val: float = -100.0, max_val: float = 100.0) -> float:
        return random.uniform(min_val, max_val)

    def _generate_str(self, length: int = 10) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def _generate_bool(self) -> bool:
        return random.choice([True, False])

    def _generate_date(self, start_year: int = 2000, end_year: int = 2024) -> str:
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime('%Y-%m-%d')

    def _generate_list(self, schema: Dict[str, Any], length: int = 3) -> List[Any]:
        return [self.generate_sample(schema) for _ in range(length)]

    def _generate_tuple(self, schema: Dict[str, Any], length: int = 3) -> Tuple[Any, ...]:
        return tuple(self.generate_sample(schema) for _ in range(length))

    def _generate_dict(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self.generate_sample(value) for key, value in schema.items()}

    def generate_sample(self, schema: Union[Dict[str, Any], str, List[Any]]) -> Any:
        if isinstance(schema, str):
            if schema in self.type_generators:
                return self.type_generators[schema]()
            raise ValueError(f"Unsupported type: {schema}")
        
        if isinstance(schema, dict):
            if 'type' in schema:
                data_type = schema['type']
                if data_type in ['list', 'tuple']:
                    length = schema.get('length', 3)
                    item_schema = schema.get('items', 'str')
                    return self.type_generators[data_type](item_schema, length)
                elif data_type == 'dict':
                    return self._generate_dict(schema.get('properties', {}))
                elif data_type in self.type_generators:
                    return self.type_generators[data_type](**{k: v for k, v in schema.items() if k != 'type'})
            else:
                return self._generate_dict(schema)
        
        raise ValueError(f"Invalid schema format: {schema}")

    def generate_samples(self, schema: Dict[str, Any], num_samples: int = 1) -> List[Any]:
        return [self.generate_sample(schema) for _ in range(num_samples)]

# 使用示例
if __name__ == "__main__":
    # 创建数据采样器实例
    sampler = DataSampler()
    
    # 示例1：简单数据类型
    simple_schema = "int"
    print("Simple integer:", sampler.generate_sample(simple_schema))
    
    # 示例2：嵌套字典
    user_schema = {
        "id": "int",
        "name": "str",
        "is_active": "bool",
        "registration_date": "date",
        "scores": {
            "type": "list",
            "items": "float",
            "length": 3
        },
        "preferences": {
            "theme": "str",
            "notifications": "bool"
        }
    }
    print("\nUser data:", sampler.generate_sample(user_schema))
    
    # 示例3：复杂嵌套结构
    complex_schema = {
        "type": "dict",
        "properties": {
            "users": {
                "type": "list",
                "length": 2,
                "items": {
                    "id": "int",
                    "name": "str",
                    "contacts": {
                        "type": "tuple",
                        "items": "str",
                        "length": 2
                    }
                }
            },
            "metadata": {
                "created_at": "date",
                "version": "float"
            }
        }
    }
    print("\nComplex nested data:", sampler.generate_sample(complex_schema))
    
    # 示例4：生成多个样本
    print("\nMultiple samples:", sampler.generate_samples(user_schema, num_samples=2))
