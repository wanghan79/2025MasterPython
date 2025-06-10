import random
import string
from datetime import datetime, timedelta

class DataSampler:
    @staticmethod
    def generate_random_data(data_type):
        """
        生成指定类型的随机数据
        """
        if data_type == 'int':
            return random.randint(0, 10000)
        elif data_type == 'float':
            return round(random.uniform(0, 100), 2)
        elif data_type == 'str':
            length = random.randint(5, 15)
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        elif data_type == 'bool':
            return random.choice([True, False])
        elif data_type == 'date':
            start_date = datetime(2000, 1, 1)
            end_date = datetime(2023, 12, 31)
            delta = end_date - start_date
            random_days = random.randint(0, delta.days)
            return (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    @staticmethod
    def generate_nested_structure(structure):
        """
        生成嵌套数据结构
        """
        if isinstance(structure, dict):
            # 处理字典类型
            result = {}
            for key, value in structure.items():
                result[key] = DataSampler.generate_nested_structure(value)
            return result
        elif isinstance(structure, list):
            # 处理列表类型
            return [DataSampler.generate_nested_structure(item) for item in structure]
        elif isinstance(structure, tuple):
            # 处理元组类型
            return tuple(DataSampler.generate_nested_structure(item) for item in structure)
        elif isinstance(structure, str) and structure in ['int', 'float', 'str', 'bool', 'date']:
            # 处理基本数据类型
            return DataSampler.generate_random_data(structure)
        else:
            return structure

    @staticmethod
    def generate_samples(sample_count, structure):
        """
        生成指定数量的样本
        """
        return [DataSampler.generate_nested_structure(structure) for _ in range(sample_count)]

    @classmethod
    def sample(cls, **kwargs):
        """
        主接口函数，根据kwargs参数生成随机样本
        """
        if 'structure' not in kwargs or 'count' not in kwargs:
            raise ValueError("Must provide 'structure' and 'count' parameters")
        
        sample_count = kwargs['count']
        structure = kwargs['structure']
        
        return cls.generate_samples(sample_count, structure)


# 使用示例
if __name__ == "__main__":
    # 示例1：生成简单的用户数据
    user_structure = {
        "id": "int",
        "name": "str",
        "is_active": "bool",
        "balance": "float",
        "register_date": "date",
        "tags": ["str"],  # 字符串列表
    }
    
    users = DataSampler.sample(structure=user_structure, count=3)
    print("生成的用户数据示例:")
    for user in users:
        print(user)
    
    # 示例2：生成嵌套更复杂的数据
    complex_structure = {
        "meta": {
            "timestamp": "date",
            "version": "str"
        },
        "data": [
            {
                "id": "int",
                "values": ("float", "float", "float"),
                "attributes": {
                    "valid": "bool",
                    "description": "str"
                }
            }
        ]
    }
    
    complex_data = DataSampler.sample(structure=complex_structure, count=2)
    print("\n生成的复杂嵌套数据示例:")
    for data in complex_data:
        print(data)