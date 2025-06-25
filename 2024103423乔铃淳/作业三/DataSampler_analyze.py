import random
import string
import datetime
from typing import Any, Union, List, Dict, Tuple, Callable
import csv
import math
from collections import defaultdict

def stats_decorator(stats: Tuple[str, ...] = ('SUM', 'AVG', 'VAR', 'RMSE'), 
                   csv_path: str = None) -> Callable:
    """
    带参数的装饰器，用于统计生成数据中的数值型数据
    :param stats: 需要计算的统计项，默认为全部统计项
    :param csv_path: CSV文件保存路径，如果为None则不保存
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # 调用原始函数生成数据
            data = func(*args, **kwargs)
            
            # 按字段路径收集所有数值型数据
            field_values = defaultdict(list)
            for sample in data:
                collect_numeric_fields(sample, field_values)
            
            # 计算并打印每个字段的统计结果
            print("\n字段级数值统计结果:")
            for field_path, values in field_values.items():
                if not values:
                    continue
                
                n = len(values)
                total = sum(values)
                mean = total / n
                squared_diffs = sum((x - mean) ** 2 for x in values)
                
                field_stats = {}
                if 'SUM' in stats:
                    field_stats['SUM'] = total
                if 'AVG' in stats:
                    field_stats['AVG'] = mean
                if 'VAR' in stats:
                    field_stats['VAR'] = squared_diffs / n
                if 'RMSE' in stats:
                    field_stats['RMSE'] = math.sqrt(squared_diffs / n)
                
                # 打印当前字段的统计结果
                print(f"\n字段: {field_path}")
                for stat, value in field_stats.items():
                    if isinstance(value, float):
                        print(f"  {stat}: {value:.4f}")
                    else:
                        print(f"  {stat}: {value}")
            
            # 保存数据到CSV
            if csv_path and data:
                save_to_csv(data, csv_path)
                print(f"\n生成的数据已保存到: {csv_path}")
            
            return data
        return wrapper
    return decorator

def collect_numeric_fields(data: Any, field_values: defaultdict, parent_path: str = "") -> None:
    """
    递归收集所有数值型字段的值
    :param data: 当前处理的数据
    :param field_values: 存储字段值的字典
    :param parent_path: 父级字段路径
    """
    if isinstance(data, (int, float)):
        # 保存数值数据
        if parent_path:
            field_values[parent_path].append(data)
        else:
            # 如果是顶层数值，添加一个通用名称
            field_values["root_value"].append(data)
    
    elif isinstance(data, dict):
        # 处理字典
        for key, value in data.items():
            new_path = f"{parent_path}.{key}" if parent_path else key
            collect_numeric_fields(value, field_values, new_path)
    
    elif isinstance(data, (list, tuple)):
        # 处理列表/元组 - 使用索引
        for i, item in enumerate(data):
            # 使用索引表示列表位置
            new_path = f"{parent_path}[{i}]"
            collect_numeric_fields(item, field_values, new_path)
    
    elif isinstance(data, datetime.date):
        # 日期类型不收集
        pass
    
    elif data is None:
        # 空值不收集
        pass
    
    else:
        # 其他类型（字符串、布尔值等）不收集
        pass

def save_to_csv(data: List[Dict], csv_path: str) -> None:
    """将嵌套字典数据保存为CSV文件"""
    # 收集所有可能的字段名（扁平化嵌套字典）
    fieldnames = set()
    for sample in data:
        for key in flatten_dict(sample).keys():
            fieldnames.add(key)
    
    # 写入CSV文件
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        
        for sample in data:
            flat_sample = flatten_dict(sample)
            writer.writerow(flat_sample)

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """将嵌套字典扁平化为单层字典"""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, (list, tuple)):
            # 将列表/元组转换为字符串表示
            items[new_key] = str(v)
        else:
            items[new_key] = v
    return items

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
    
    @stats_decorator(stats=('SUM', 'AVG', 'VAR', 'RMSE'), csv_path=None)
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
    
    # 生成10个样本（包含统计和CSV保存）
    users = sampler.generate_data(
        num_samples=10,
        structure=user_schema,
        csv_path="/datapool/home/gongjianting/EnzyCom_zero_shot/homework/1/user_data.csv"  # 设置CSV文件保存路径
    )
    
   