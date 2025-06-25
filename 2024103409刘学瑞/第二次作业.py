import random
import string
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime, timedelta

class DataGenerator:
    """数据生成器类，用于生成各种类型的随机数据"""
    
    @staticmethod
    def generate_int(value: Dict) -> int:
        """生成整数"""
        return random.randint(value["datarange"][0], value["datarange"][1])
    
    @staticmethod
    def generate_float(value: Dict) -> float:
        """生成浮点数"""
        return random.uniform(value["datarange"][0], value["datarange"][1])
    
    @staticmethod
    def generate_str(value: Dict) -> str:
        """生成字符串"""
        length = value.get("len", 10)
        return ''.join(random.SystemRandom().choice(value["datarange"]) for _ in range(length))
    
    @staticmethod
    def generate_bool(value: Dict) -> bool:
        """生成布尔值"""
        return random.choice([True, False])
    
    @staticmethod
    def generate_datetime(value: Dict) -> datetime:
        """生成日期时间"""
        start_date = value.get("start_date", datetime(2000, 1, 1))
        end_date = value.get("end_date", datetime(2024, 12, 31))
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_days = random.randrange(days_between_dates)
        return start_date + timedelta(days=random_days)
    
    @staticmethod
    def generate_list(value: Dict) -> List:
        """生成列表"""
        size = value.get("size", 5)
        return [DataGenerator.generate_value(value["subNodes"]) for _ in range(size)]
    
    @staticmethod
    def generate_tuple(value: Dict) -> Tuple:
        """生成元组"""
        size = value.get("size", 5)
        return tuple(DataGenerator.generate_value(value["subNodes"]) for _ in range(size))
    
    @staticmethod
    def generate_dict(value: Dict) -> Dict:
        """生成字典"""
        return DataGenerator.recursive_sampling(value["subNodes"])
    
    @staticmethod
    def generate_value(value: Dict) -> Any:
        """根据类型生成对应的值"""
        type_handlers = {
            "int": DataGenerator.generate_int,
            "float": DataGenerator.generate_float,
            "str": DataGenerator.generate_str,
            "bool": DataGenerator.generate_bool,
            "datetime": DataGenerator.generate_datetime,
            "list": DataGenerator.generate_list,
            "tuple": DataGenerator.generate_tuple,
            "dict": DataGenerator.generate_dict
        }
        
        data_type = value.get("type", "str")
        if data_type not in type_handlers:
            raise ValueError(f"不支持的数据类型: {data_type}")
            
        return type_handlers[data_type](value)

    @staticmethod
    def recursive_sampling(node: Dict) -> Dict:
        """递归采样函数，用于处理嵌套结构"""
        sample = {}
        for key, value in node.items():
            if isinstance(value, dict):
                if "subNodes" in value:
                    sample[key] = DataGenerator.recursive_sampling(value["subNodes"])
                else:
                    sample[key] = DataGenerator.generate_value(value)
            else:
                sample[key] = value
        return sample

def Sampling(**kwargs) -> List[Any]:
    """
    采样函数，根据给定的参数生成相应的数据。

    Args:
        **kwargs: 包含数据类型、范围和结构的键值对。

    Returns:
        生成的样本数据列表。
    """
    samples = []
    for key, value in kwargs.items():
        if isinstance(value, dict) and "subNodes" in value:
            samples.append(DataGenerator.recursive_sampling(value["subNodes"]))
        else:
            samples.append(DataGenerator.generate_value(value))
    return samples

# 示例用法
if __name__ == "__main__":
    # 定义三层嵌套的数据结构
    data_structure = {
        "user": {
            "subNodes": {
                "id": {"type": "int", "datarange": (1, 1000)},
                "name": {"type": "str", "datarange": string.ascii_uppercase, "len": 10},
                "profile": {
                    "subNodes": {
                        "age": {"type": "int", "datarange": (18, 80)},
                        "email": {"type": "str", "datarange": string.ascii_lowercase, "len": 20},
                        "address": {
                            "subNodes": {
                                "city": {"type": "str", "datarange": ["北京", "上海", "广州", "深圳"], "len": 1},
                                "street": {"type": "str", "datarange": string.ascii_uppercase, "len": 15},
                                "zipcode": {"type": "int", "datarange": (100000, 999999)}
                            }
                        }
                    }
                },
                "orders": {
                    "type": "list",
                    "size": 3,
                    "subNodes": {
                        "order_id": {"type": "int", "datarange": (1000, 9999)},
                        "amount": {"type": "float", "datarange": (100.0, 1000.0)},
                        "items": {
                            "type": "tuple",
                            "size": 2,
                            "subNodes": {
                                "product_id": {"type": "int", "datarange": (1, 100)},
                                "quantity": {"type": "int", "datarange": (1, 10)}
                            }
                        }
                    }
                }
            }
        }
    }

    # 生成样本数据
    samples = Sampling(**data_structure)

    # 打印样本数据
    import json
    print(json.dumps(samples, ensure_ascii=False, indent=2))
