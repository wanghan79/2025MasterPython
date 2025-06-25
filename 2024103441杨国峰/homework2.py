import random
import string
from datetime import datetime, timedelta

class DataSampler:
    def __init__(self, random_generator=None):
        """
        初始化数据采样器
        
        Args:
            random_generator: 可选的随机数生成器，默认使用random模块
        """
        self.random_generator = random_generator or random
    
    def random_value(self, data_type, data_range=None):
        """
        根据指定的数据类型和参数生成随机值
        
        Args:
            data_type: 数据类型（int, float, str, bool, list, tuple, dict, date等）
            data_range: 数据范围参数
            
        Returns:
            生成的随机值
        """
        # 整数类型
        if data_type == int:
            return self.random_generator.randint(data_range[0], data_range[1])
        
        # 浮点数类型
        elif data_type == float:
            return self.random_generator.uniform(data_range[0], data_range[1])
        
        # 字符串类型
        elif data_type == str:
            if isinstance(data_range, int):
                # 如果data_range是整数，生成指定长度的随机字符串
                return ''.join(self.random_generator.choices(string.ascii_letters, k=data_range))
            elif isinstance(data_range, list):
                # 如果data_range是列表，从列表中随机选择一个字符串
                return self.random_generator.choice(data_range)
        
        # 布尔类型
        elif data_type == bool:
            return self.random_generator.choice([True, False])
        
        # 列表类型
        elif data_type == list:
            return [self.random_value(data_range['type'], data_range.get('range')) 
                    for _ in range(data_range['length'])]
        
        # 元组类型
        elif data_type == tuple:
            return tuple(self.random_value(data_range['type'], data_range.get('range')) 
                         for _ in range(data_range['length']))
        
        # 字典类型
        elif data_type == dict:
            return self.generate_structure(data_range)
        
        # 日期类型
        elif data_type == 'date':
            start_date = data_range[0]
            end_date = data_range[1]
            random_days = self.random_generator.randint(0, (end_date - start_date).days)
            return start_date + timedelta(days=random_days)
        
        # 不支持的类型
        else:
            return None
    
    def generate_structure(self, structure):
        """
        根据指定的结构生成完整的数据样本
        
        Args:
            structure: 数据结构定义
            
        Returns:
            生成的数据结构
        """
        if isinstance(structure, dict):
            result = {}
            for key, value in structure.items():
                if isinstance(value, dict):
                    data_type = value.get('type')
                    data_range = value.get('range')
                    subs = value.get('subs', [])
                    
                    if isinstance(subs, list) and subs:
                        result[key] = [self.generate_structure(sub) for sub in subs]
                    else:
                        result[key] = self.random_value(data_type, data_range)
                else:
                    result[key] = value
            return result
        else:
            raise ValueError("不支持的结构类型")
    
    def generate_samples(self, structure, num_samples=1):
        """
        生成多个数据样本
        
        Args:
            structure: 数据结构定义
            num_samples: 要生成的样本数量
            
        Returns:
            生成的样本列表
        """
        return [self.generate_structure(structure) for _ in range(num_samples)]


# 示例用法
if __name__ == "__main__":
    # 定义数据结构
    data_structure = {
        'user_id': {'type': int, 'range': (1000, 9999)},
        'username': {'type': str, 'range': 8},
        'is_active': {'type': bool},
        'score': {'type': float, 'range': (0.0, 100.0)},
        'tags': {
            'type': list, 
            'range': {
                'type': str, 
                'range': 5, 
                'length': 3
            }
        },
        'login_history': {
            'type': dict,
            'subs': [
                {'last_login': {'type': 'date', 'range': [datetime(2023, 1, 1), datetime(2023, 12, 31)]}},
                {'login_count': {'type': int, 'range': (0, 100)}}
            ]
        }
    }
    
    # 创建采样器并生成样本
    sampler = DataSampler()
    samples = sampler.generate_samples(data_structure, num_samples=3)
    
    # 打印生成的样本
    for i, sample in enumerate(samples, 1):
        print(f"样本 {i}:")
        print(sample)
        print()
