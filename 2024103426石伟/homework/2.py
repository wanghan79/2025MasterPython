import random
import string
import datetime
from typing import Any, Dict, List, Union, Callable

class DataSampler:
    def __init__(self, seed: int = None):
        """
        初始化随机数据生成器
        :param seed: 随机种子（可选）
        """
        if seed is not None:
            random.seed(seed)
        
        # 类型生成器映射
        self.type_generators = {
            int: self._generate_int,
            float: self._generate_float,
            str: self._generate_str,
            bool: self._generate_bool,
            datetime.date: self._generate_date,
            list: self._generate_list,
            tuple: self._generate_tuple,
            dict: self._generate_dict,
        }
    
    def generate(self, num_samples: int, **kwargs) -> List[Dict[str, Any]]:
        """
        生成指定数量的随机样本
        :param num_samples: 样本数量
        :param kwargs: 数据结构定义
        :return: 生成的样本列表
        """
        return [self._generate_sample(kwargs) for _ in range(num_samples)]
    
    def _generate_sample(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """生成单个样本"""
        sample = {}
        for key, spec in structure.items():
            sample[key] = self._generate_value(spec)
        return sample
    
    def _generate_value(self, spec: Any) -> Any:
        """根据规范生成随机值"""
        # 如果是类型直接指定
        if isinstance(spec, type):
            return self.type_generators[spec]()
        
        # 如果是带有参数的生成器规范
        if isinstance(spec, dict):
            value_type = spec.get('type', None)
            
            # 处理自定义参数的基本类型
            if value_type in (int, float, str, bool, datetime.date):
                return self.type_generators[value_type](**spec)
            
            # 处理嵌套结构
            if value_type in (list, tuple, dict):
                return self.type_generators[value_type](spec)
        
        # 如果是自定义生成函数
        if callable(spec):
            return spec()
        
        # 如果是固定值
        return spec
    
    # ===== 基本类型生成器 =====
    def _generate_int(self, min_val: int = 0, max_val: int = 100) -> int:
        return random.randint(min_val, max_val)
    
    def _generate_float(self, min_val: float = 0.0, max_val: float = 1.0, 
                       precision: int = 2) -> float:
        value = random.uniform(min_val, max_val)
        return round(value, precision)
    
    def _generate_str(self, length: int = 8, prefix: str = "", 
                     charset: str = string.ascii_letters + string.digits) -> str:
        if length <= len(prefix):
            return prefix
        random_part = ''.join(random.choices(charset, k=length - len(prefix)))
        return prefix + random_part
    
    def _generate_bool(self) -> bool:
        return random.choice([True, False])
    
    def _generate_date(self, start: datetime.date = datetime.date(2000, 1, 1),
                      end: datetime.date = datetime.date.today()) -> datetime.date:
        days_between = (end - start).days
        random_days = random.randint(0, days_between)
        return start + datetime.timedelta(days=random_days)
    
    # ===== 嵌套结构生成器 =====
    def _generate_list(self, spec: dict) -> list:
        size = spec.get('size', random.randint(1, 5))
        element_spec = spec.get('element', str)  # 默认为字符串列表
        
        return [self._generate_value(element_spec) for _ in range(size)]
    
    def _generate_tuple(self, spec: dict) -> tuple:
        elements_spec = spec.get('elements', [str, int])  # 默认为(字符串, 整数)
        return tuple(self._generate_value(elem_spec) for elem_spec in elements_spec)
    
    def _generate_dict(self, spec: dict) -> dict:
        structure = spec.get('structure', {'key': str, 'value': int})  # 默认字典结构
        return self._generate_sample(structure)

# ===== 使用示例 =====
if __name__ == "__main__":
    # 创建数据生成器（可设置随机种子保证结果可复现）
    sampler = DataSampler(seed=42)
    
    # 定义复杂嵌套数据结构
    user_data_structure = {
        "user_id": int,
        "username": {
            "type": str,
            "prefix": "user_",
            "length": 10
        },
        "profile": {
            "type": dict,
            "structure": {
                "full_name": str,
                "age": {
                    "type": int,
                    "min_val": 18,
                    "max_val": 65
                },
                "birthdate": datetime.date,
                "preferences": {
                    "type": list,
                    "element": {
                        "type": str,
                        "prefix": "pref_",
                        "length": 6
                    },
                    "size": 3
                }
            }
        },
        "contact_info": {
            "type": tuple,
            "elements": [
                {"type": str, "prefix": "email_", "length": 15},
                {"type": str, "prefix": "phone_", "length": 12}
            ]
        },
        "is_active": bool,
        "metadata": {
            "type": dict,
            "structure": {
                "created_at": datetime.date,
                "last_login": datetime.date,
                "login_count": int
            }
        }
    }
    
    # 生成5个样本
    samples = sampler.generate(5, **user_data_structure)
    
    # 打印生成的样本
    for i, sample in enumerate(samples, 1):
        print(f"\n=== 用户样本 #{i} ===")
        print(sample)
