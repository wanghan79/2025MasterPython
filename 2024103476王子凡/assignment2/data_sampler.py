import random
import string
from datetime import datetime, timedelta
import numpy as np

class DataSampler:
    @staticmethod
    def _generate_random_string(length=8):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def _generate_random_date(start_date=datetime(2000, 1, 1), end_date=datetime(2023, 12, 31)):
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        return start_date + timedelta(days=random_number_of_days)
    
    @staticmethod
    def _generate_value(data_type):
        if data_type == int:
            return random.randint(-1000, 1000)
        elif data_type == float:
            return round(random.uniform(-1000, 1000), 2)
        elif data_type == str:
            return DataSampler._generate_random_string()
        elif data_type == bool:
            return random.choice([True, False])
        elif data_type == datetime:
            return DataSampler._generate_random_date()
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
    
    @staticmethod
    def _generate_structure(structure):
        if isinstance(structure, (list, tuple)):
            return type(structure)(DataSampler._generate_structure(item) for item in structure)
        elif isinstance(structure, dict):
            return {k: DataSampler._generate_structure(v) for k, v in structure.items()}
        elif isinstance(structure, type):
            return DataSampler._generate_value(structure)
        else:
            return structure
    
    @staticmethod
    def generate_samples(structure, num_samples=1):
        """
        生成指定结构的随机样本
        
        参数:
        structure: 定义数据结构的模板
        num_samples: 需要生成的样本数量
        
        返回:
        生成的样本列表
        """
        return [DataSampler._generate_structure(structure) for _ in range(num_samples)]

# 使用示例
if __name__ == "__main__":
    # 示例1：生成用户数据
    user_structure = {
        "id": int,
        "name": str,
        "age": int,
        "is_active": bool,
        "scores": [float, float, float],
        "address": {
            "city": str,
            "zip_code": str
        },
        "created_at": datetime
    }
    
    # 生成3个用户样本
    users = DataSampler.generate_samples(user_structure, 3)
    print("生成的用户数据示例：")
    for user in users:
        print(user)
    
    # 示例2：生成嵌套列表数据
    nested_list_structure = [
        [int, float],
        [str, bool],
        {"key1": int, "key2": [float, float]}
    ]
    
    # 生成2个嵌套列表样本
    nested_samples = DataSampler.generate_samples(nested_list_structure, 2)
    print("\n生成的嵌套列表数据示例：")
    for sample in nested_samples:
        print(sample) 