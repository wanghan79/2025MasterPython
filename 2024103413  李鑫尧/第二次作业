import random
import string
from datetime import datetime, timedelta


class DataSampler:
    def __init__(self, random_generator=None):
        """
        初始化数据采样器。

        Args:
            random_generator: 可选的随机数生成器，默认为random.SystemRandom
        """
        self.random_generator = random_generator or random.SystemRandom()
    
    def generate_value(self, value_def):
        """
        根据值定义生成相应的随机值。

        Args:
            value_def: 包含数据类型和参数的字典或直接值

        Returns:
            生成的随机值。
        """
        if not isinstance(value_def, dict):
            return value_def
        
        if "type" not in value_def:
            return value_def
        
        data_type = value_def["type"]
        params = {k: v for k, v in value_def.items() if k != "type"}
        
        # 处理不同类型的数据生成
        if data_type == "int":
            return self.random_generator.randint(params["datarange"][0], params["datarange"][1])
        elif data_type == "float":
            return round(self.random_generator.uniform(params["datarange"][0], params["datarange"][1]), 2)
        elif data_type == "str":
            length = params.get("len", 8)
            charset = params.get("datarange", string.ascii_letters + string.digits)
            return ''.join(self.random_generator.choice(charset) for _ in range(length))
        elif data_type == "bool":
            return self.random_generator.choice([True, False])
        elif data_type == "date":
            start = datetime.strptime(params["datarange"][0], "%Y-%m-%d")
            end = datetime.strptime(params["datarange"][1], "%Y-%m-%d")
            delta = end - start
            random_days = self.random_generator.randint(0, delta.days)
            return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")
        elif data_type == "datetime":
            start = datetime.strptime(params["datarange"][0], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(params["datarange"][1], "%Y-%m-%d %H:%M:%S")
            delta = end - start
            random_seconds = self.random_generator.randint(0, int(delta.total_seconds()))
            return (start + timedelta(seconds=random_seconds)).strftime("%Y-%m-%d %H:%M:%S")
        elif data_type == "list":
            list_length = self.random_generator.randint(params.get("min_len", 1), params.get("max_len", 5))
            element_def = params["elements"]
            return [self.generate_value(element_def) for _ in range(list_length)]
        elif data_type == "tuple":
            tuple_length = self.random_generator.randint(params.get("min_len", 1), params.get("max_len", 5))
            element_def = params["elements"]
            return tuple(self.generate_value(element_def) for _ in range(tuple_length))
        elif data_type == "dict":
            return {k: self.generate_value(v) for k, v in params["subs"].items()}
        elif data_type == "choice":
            return self.random_generator.choice(params["options"])
        else:
            return None

    def generate_sample(self, structure):
        """
        根据给定的结构生成单个样本

        Args:
            structure: 包含数据结构定义的字典

        Returns:
            生成的样本数据字典。
        """
        sample = {}
        for key, value_def in structure.items():
            sample[key] = self.generate_value(value_def)
        return sample

    def generate_samples(self, n, **structure):
        """
        生成多个样本数据
        
        Args:
            n: 样本数量
            **structure: 包含数据结构定义的键值对
        
        Returns:
            包含n个样本的列表
        """
        return [self.generate_sample(structure) for _ in range(n)]


if __name__ == "__main__":
    sampler = DataSampler()
    
    # 定义数据结构
    data_structure = {
        "user_id": {
            "type": "int",
            "datarange": (100000, 999999)
        },
        "username": {
            "type": "str",
            "len": 8,
            "datarange": string.ascii_lowercase
        },
        "is_active": {
            "type": "bool"
        },
        "registration_date": {
            "type": "date",
            "datarange": ("2020-01-01", "2023-12-31")
        },
        "last_login": {
            "type": "datetime",
            "datarange": ("2023-01-01 00:00:00", "2023-12-31 23:59:59")
        },
        "profile": {
            "type": "dict",
            "subs": {
                "age": {
                    "type": "int",
                    "datarange": (18, 99)
                },
                "gender": {
                    "type": "choice",
                    "options": ["Male", "Female", "Other"]
                }
            }
        },
        "scores": {
            "type": "list",
            "min_len": 3,
            "max_len": 7,
            "elements": {
                "type": "float",
                "datarange": (0.0, 100.0)
            }
        },
        "preferences": {
            "type": "dict",
            "subs": {
                "theme": {
                    "type": "choice",
                    "options": ["Light", "Dark", "System"]
                },
                "notifications": {
                    "type": "bool"
                }
            }
        },
        "status": "active"  
    }
    
    # 生成5个样本
    samples = sampler.generate_samples(5, **data_structure)
    
    # 打印样本
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}:")
        print(sample)
        print("-" * 50)
