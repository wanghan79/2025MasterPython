import random
import string
import datetime
from typing import Dict, List, Tuple, Any, Union

class DataSampler:
    """
    随机数据生成器，生成结构化的模拟数据
    支持多种数据类型（int、float、str、bool、date、list、tuple、dict）
    支持嵌套数据结构生成
    """
    
    @staticmethod
    def generate_int(min_val: int = 0, max_val: int = 100) -> int:
        """生成随机整数"""
        return random.randint(min_val, max_val)
    
    @staticmethod
    def generate_float(min_val: float = 0.0, max_val: float = 1.0, precision: int = 2) -> float:
        """生成随机浮点数"""
        return round(random.uniform(min_val, max_val), precision)
    
    @staticmethod
    def generate_str(length: int = 10) -> str:
        """生成随机字符串"""
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    
    @staticmethod
    def generate_bool() -> bool:
        """生成随机布尔值"""
        return random.choice([True, False])
    
    @staticmethod
    def generate_date(start_year: int = 2000, end_year: int = 2023) -> datetime.date:
        """生成随机日期"""
        year = random.randint(start_year, end_year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # 简化处理，避免日期无效问题
        return datetime.date(year, month, day)
    
    def generate_value(self, data_type: Dict[str, Any]) -> Any:
        """根据指定类型生成对应的随机值"""
        type_name = data_type.get("type", "str")
        
        if type_name == "int":
            min_val = data_type.get("min", 0)
            max_val = data_type.get("max", 100)
            return self.generate_int(min_val, max_val)
        
        elif type_name == "float":
            min_val = data_type.get("min", 0.0)
            max_val = data_type.get("max", 1.0)
            precision = data_type.get("precision", 2)
            return self.generate_float(min_val, max_val, precision)
        
        elif type_name == "str":
            length = data_type.get("length", 10)
            return self.generate_str(length)
        
        elif type_name == "bool":
            return self.generate_bool()
        
        elif type_name == "date":
            start_year = data_type.get("start_year", 2000)
            end_year = data_type.get("end_year", 2023)
            return self.generate_date(start_year, end_year)
        
        elif type_name == "list":
            return self.generate_list(data_type)
        
        elif type_name == "tuple":
            return tuple(self.generate_list(data_type))
        
        elif type_name == "dict":
            return self.generate_dict(data_type)
        
        else:
            return None
    
    def generate_list(self, data_type: Dict[str, Any]) -> List[Any]:
        """生成随机列表"""
        item_type = data_type.get("item_type", {"type": "str"})
        length = data_type.get("length", 5)
        min_length = data_type.get("min_length", length)
        max_length = data_type.get("max_length", length)
        
        actual_length = random.randint(min_length, max_length)
        return [self.generate_value(item_type) for _ in range(actual_length)]
    
    def generate_dict(self, data_type: Dict[str, Any]) -> Dict[str, Any]:
        """生成随机字典"""
        schema = data_type.get("schema", {})
        result = {}
        
        for key, value_type in schema.items():
            result[key] = self.generate_value(value_type)
            
        return result
    
    def generate_sample(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """生成单个数据样本"""
        return self.generate_dict({"type": "dict", "schema": schema})
    
    def generate_samples(self, schema: Dict[str, Any], count: int = 1) -> List[Dict[str, Any]]:
        """生成多个数据样本"""
        return [self.generate_sample(schema) for _ in range(count)]


# 使用示例
def main():
    # 实例化数据生成器
    sampler = DataSampler()
    
    # 定义数据结构模式
    user_schema = {
        "id": {"type": "int", "min": 1000, "max": 9999},
        "name": {"type": "str", "length": 8},
        "is_active": {"type": "bool"},
        "registration_date": {"type": "date", "start_year": 2020, "end_year": 2023},
        "score": {"type": "float", "min": 0, "max": 100, "precision": 1},
        "tags": {"type": "list", "item_type": {"type": "str", "length": 5}, "min_length": 2, "max_length": 5},
        "preferences": {
            "type": "dict",
            "schema": {
                "theme": {"type": "str", "length": 6},
                "notifications": {"type": "bool"},
                "favorite_numbers": {"type": "list", "item_type": {"type": "int", "min": 1, "max": 100}, "length": 3}
            }
        },
        "coordinates": {"type": "tuple", "item_type": {"type": "float", "min": -90, "max": 90}, "length": 2}
    }
    
    # 生成样本数据
    print("生成单个用户样本:")
    user = sampler.generate_sample(user_schema)
    print(user)
    print("\n")
    
    print("生成多个用户样本:")
    users = sampler.generate_samples(user_schema, 3)
    for i, user in enumerate(users, 1):
        print(f"用户 {i}:")
        print(user)
        print()

if __name__ == "__main__":
    main() 