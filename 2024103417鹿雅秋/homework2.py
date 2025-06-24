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

    def _generate_int(self, min_val: int = 0, max_val: int = 100) -> int:
        return random.randint(min_val, max_val)

    def _generate_float(self, min_val: float = 0.0, max_val: float = 100.0) -> float:
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
            # 处理基本数据类型
            if schema in self.type_generators:
                return self.type_generators[schema]()
        elif isinstance(schema, dict):
            if 'type' in schema:
                # 处理带有特定参数的数据类型
                data_type = schema['type']
                if data_type in self.type_generators:
                    params = {k: v for k, v in schema.items() if k != 'type'}
                    return self.type_generators[data_type](**params)
            else:
                # 处理嵌套字典
                return self._generate_dict(schema)
        elif isinstance(schema, list):
            # 处理列表类型
            return [self.generate_sample(item) for item in schema]
        raise ValueError(f"Unsupported schema type: {schema}")

    def generate_samples(self, schema: Dict[str, Any], num_samples: int = 1) -> List[Any]:
        """生成多个数据样本"""
        return [self.generate_sample(schema) for _ in range(num_samples)]

# 示例用法
def main():
    sampler = DataSampler()
    
    # 示例1：生成简单的用户数据
    user_schema = {
        "id": "int",
        "name": "str",
        "age": {"type": "int", "min_val": 18, "max_val": 80},
        "is_active": "bool",
        "registration_date": "date"
    }
    
    # 示例2：生成嵌套的复杂数据
    complex_schema = {
        "user": {
            "id": "int",
            "profile": {
                "name": "str",
                "age": {"type": "int", "min_val": 18, "max_val": 80},
                "scores": {"type": "list", "schema": "float", "length": 3}
            }
        },
        "orders": {"type": "list", "schema": {
            "order_id": "str",
            "amount": {"type": "float", "min_val": 10.0, "max_val": 1000.0},
            "items": {"type": "tuple", "schema": "str", "length": 2}
        }, "length": 2}
    }

    # 生成样本
    print("简单用户数据示例：")
    print(sampler.generate_samples(user_schema, 2))
    print("\n复杂嵌套数据示例：")
    print(sampler.generate_samples(complex_schema, 1))

if __name__ == "__main__":
    main()
