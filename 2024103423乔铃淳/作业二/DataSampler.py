import random
import string
import datetime
from typing import Any, Union, List, Dict, Tuple

class DataSampler:
    def __init__(self, seed: int = None):
        """
        初始化数据生成器
        :param seed: 随机种子，用于复现结果
        """
        self.random = random.Random(seed)
        self.type_handlers = {
            int: self._gen_int,
            float: self._gen_float,
            str: self._gen_str,
            bool: self._gen_bool,
            datetime.date: self._gen_date,
            list: self._gen_list,
            tuple: self._gen_tuple,
            dict: self._gen_dict
        }
    
    def _gen_int(self, **kwargs) -> int:
        """生成随机整数"""
        min_val = kwargs.get('min_val', 0)
        max_val = kwargs.get('max_val', 100)
        return self.random.randint(min_val, max_val)
    
    def _gen_float(self, **kwargs) -> float:
        """生成随机浮点数"""
        min_val = kwargs.get('min_val', 0.0)
        max_val = kwargs.get('max_val', 100.0)
        return self.random.uniform(min_val, max_val)
    
    def _gen_str(self, **kwargs) -> str:
        """生成随机字符串"""
        min_len = kwargs.get('min_len', 5)
        max_len = kwargs.get('max_len', 15)
        length = self.random.randint(min_len, max_len)
        return ''.join(self.random.choices(string.ascii_letters + string.digits, k=length))
    
    def _gen_bool(self, **kwargs) -> bool:
        """生成随机布尔值"""
        return self.random.choice([True, False])
    
    def _gen_date(self, **kwargs) -> datetime.date:
        """生成随机日期"""
        start = kwargs.get('start', datetime.date(2000, 1, 1))
        end = kwargs.get('end', datetime.date(2023, 12, 31))
        delta = (end - start).days
        random_days = self.random.randint(0, delta)
        return start + datetime.timedelta(days=random_days)
    
    def _gen_list(self, schema: Any, **kwargs) -> list:
        """生成随机列表"""
        size = kwargs.get('size', self.random.randint(1, 5))
        return [self.generate_sample(schema) for _ in range(size)]
    
    def _gen_tuple(self, schema: Any, **kwargs) -> tuple:
        """生成随机元组"""
        return tuple(self._gen_list(schema, **kwargs))
    
    def _gen_dict(self, schema: Dict[str, Any], **kwargs) -> dict:
        """生成随机字典"""
        return {key: self.generate_sample(value) for key, value in schema.items()}
    
    def generate_sample(self, schema: Any) -> Any:
        """
        根据给定的模式生成单个随机样本
        :param schema: 描述数据结构的模式
        :return: 生成的随机数据
        """
        # 处理嵌套模式定义
        if isinstance(schema, dict) and 'type' in schema:
            schema_type = schema['type']
            params = {k: v for k, v in schema.items() if k != 'type'}
            return self.type_handlers[schema_type](**params)
        
        # 处理直接类型定义
        elif isinstance(schema, type) and schema in self.type_handlers:
            return self.type_handlers[schema]()
        
        # 处理列表模式
        elif isinstance(schema, list) and len(schema) == 1:
            return self._gen_list(schema[0])
        
        # 处理元组模式
        elif isinstance(schema, tuple) and len(schema) == 1:
            return self._gen_tuple(schema[0])
        
        # 处理固定结构字典
        elif isinstance(schema, dict):
            return self._gen_dict(schema)
        
        # 处理复合结构列表/元组
        elif isinstance(schema, (list, tuple)):
            return [self.generate_sample(item) for item in schema]
        
        else:
            raise ValueError(f"不支持的schema类型: {type(schema).__name__}")
    
    def generate_data(self, **kwargs) -> Union[List, Dict]:
        """
        生成结构化随机数据样本集
        :param kwargs: 必须包含 'num_samples' 和 'structure'
        :return: 生成的样本列表
        """
        num_samples = kwargs.get('num_samples', 1)
        structure = kwargs.get('structure')
        
        if structure is None:
            raise ValueError("必须提供'structure'参数定义数据结构")
        
        return [self.generate_sample(structure) for _ in range(num_samples)]


# 使用示例
if __name__ == "__main__":
    # 创建数据生成器
    sampler = DataSampler(seed=42)
    
    # 定义复杂嵌套数据结构
    user_schema = {
        "user_id": int,
        "username": str,
        "email": str,
        "is_active": bool,
        "created_at": {"type": datetime.date},
        "preferences": {
            "theme": str,
            "notifications": bool,
            "language": str
        },
        "login_history": [
            {
                "timestamp": {"type": datetime.date},
                "ip_address": str
            }
        ],
        "scores": (float,),
        "metadata": {
            "internal_id": str,
            "tags": [str]
        }
    }
    
    # 生成10个样本
    users = sampler.generate_data(
        num_samples=10,
        structure=user_schema
    )
    
    # 打印第一个样本
    import pprint
    print("生成的第一个用户样本:")
    pprint.pprint(users[0], depth=4, width=100, compact=True)