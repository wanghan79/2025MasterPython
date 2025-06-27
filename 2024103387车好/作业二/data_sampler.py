import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union
import json


class DataSampler:
    """
    结构化随机数据生成器

    功能:
    - 动态生成嵌套的字典、列表、元组等数据结构
    - 生成结构化的随机测试数据
    - 支持多种数据类型(int、float、str、bool、date、list、tuple、dict)
    """

    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def generate_random_value(self, data_type: Union[str, type, Dict]) -> Any:
        """
        生成指定类型的随机值

        参数:
            data_type: 可以是类型名称(str)、类型对象(type)或复杂结构定义(dict)
        """
        # 如果是字典结构定义，则检查是否是复杂类型定义
        if isinstance(data_type, dict):
            if 'type' in data_type or 'structure' in data_type:
                return self.generate_complex_value(data_type)
            else:
                # 普通字典，递归生成每个值
                return {k: self.generate_random_value(v) for k, v in data_type.items()}

        if isinstance(data_type, str):
            data_type = data_type.lower()

        if data_type in (int, 'int'):
            return self.random.randint(-1000, 1000)
        elif data_type in (float, 'float'):
            return self.random.uniform(-100.0, 100.0)
        elif data_type in (str, 'str'):
            length = self.random.randint(5, 15)
            return ''.join(self.random.choices(string.ascii_letters + string.digits, k=length))
        elif data_type in (bool, 'bool'):
            return self.random.choice([True, False])
        elif data_type in (datetime, 'date', 'datetime'):
            start = datetime(2000, 1, 1)
            end = datetime(2030, 12, 31)
            return start + timedelta(days=self.random.randint(0, (end - start).days))
        elif data_type in (list, 'list'):
            return []
        elif data_type in (tuple, 'tuple'):
            return ()
        elif data_type in (dict, 'dict'):
            return {}
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def generate_complex_value(self, definition: Dict[str, Any]) -> Any:
        """
        生成复杂类型的值(列表、元组、嵌套字典等)
        """
        data_type = definition.get('type', 'dict')

        if data_type in ('list', 'array'):
            # 生成列表
            length = definition.get('length', 5)
            element_type = definition.get('element_type', 'int')
            return [self.generate_random_value(element_type) for _ in range(length)]
        elif data_type in ('tuple',):
            # 生成元组
            length = definition.get('length', 5)
            element_type = definition.get('element_type', 'int')
            return tuple(self.generate_random_value(element_type) for _ in range(length))
        elif data_type in ('dict', 'object'):
            # 生成嵌套字典
            nested_structure = definition.get('structure', {})
            return self.generate_sample(nested_structure)
        else:
            raise ValueError(f"Unsupported complex type: {data_type}")

    def generate_sample(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据结构定义生成一个随机样本

        参数:
            structure: 定义数据结构,例如:
                {
                    "name": "str",
                    "age": "int",
                    "scores": {
                        "type": "list",
                        "length": 5,
                        "element_type": "int"
                    },
                    "address": {
                        "type": "dict",
                        "structure": {
                            "street": "str",
                            "number": "int"
                        }
                    }
                }
        """
        sample = {}
        for key, value in structure.items():
            try:
                sample[key] = self.generate_random_value(value)
            except ValueError as e:
                raise ValueError(f"Error generating value for key '{key}': {str(e)}")
        return sample

    def generate_samples(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """
        生成多个样本

        参数:
            count: 样本数量
            kwargs: 结构定义,例如:
                name="str",
                age="int",
                scores={
                    "type": "list",
                    "length": 5,
                    "element_type": "int"
                }
        """
        return [self.generate_sample(kwargs) for _ in range(count)]

    @staticmethod
    def print_samples(samples: List[Dict[str, Any]], indent: int = 2) -> None:
        """
        美观打印生成的样本
        """
        print(json.dumps(samples, indent=indent, default=str))


def demonstrate_data_sampler():
    """
    演示DataSampler的功能
    """
    sampler = DataSampler(seed=42)  # 固定随机种子以便重现结果

    # 简单样本
    print("简单样本示例:")
    simple_samples = sampler.generate_samples(
        3,
        name="str",
        age="int",
        is_active="bool"
    )
    sampler.print_samples(simple_samples)

    # 复杂嵌套样本
    print("\n复杂嵌套样本示例:")
    complex_samples = sampler.generate_samples(
        2,
        user_id="int",
        profile={
            "type": "dict",
            "structure": {
                "first_name": "str",
                "last_name": "str",
                "birth_date": "date",
                "address": {
                    "type": "dict",
                    "structure": {
                        "street": "str",
                        "city": "str",
                        "zip_code": "int"
                    }
                }
            }
        },
        preferences={
            "type": "list",
            "length": 3,
            "element_type": "str"
        },
        transaction_history={
            "type": "list",
            "length": 2,
            "element_type": {
                "type": "dict",
                "structure": {
                    "date": "date",
                    "amount": "float",
                    "items": {
                        "type": "list",
                        "length": 3,
                        "element_type": {
                            "type": "dict",
                            "structure": {
                                "product_id": "int",
                                "name": "str",
                                "price": "float"
                            }
                        }
                    }
                }
            }
        }
    )
    sampler.print_samples(complex_samples)

    # 混合类型样本
    print("\n混合类型样本示例:")
    mixed_samples = sampler.generate_samples(
        2,
        id="int",
        metadata={
            "type": "tuple",
            "length": 4,
            "element_type": {
                "type": "dict",
                "structure": {
                    "key": "str",
                    "value": "float",
                    "tags": {
                        "type": "list",
                        "length": 2,
                        "element_type": "str"
                    }
                }
            }
        },
        matrix={
            "type": "list",
            "length": 3,
            "element_type": {
                "type": "tuple",
                "length": 3,
                "element_type": "int"
            }
        }
    )
    sampler.print_samples(mixed_samples)


if __name__ == "__main__":
    demonstrate_data_sampler()
