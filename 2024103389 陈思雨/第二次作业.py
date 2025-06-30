import random
import string
from datetime import datetime, timedelta

class DataSampler:
    def __init__(self):
        self.samples = []
    
    def random_value(self, data_type, data_range=None):
        """生成指定类型的随机值"""
        if data_type == int:
            return random.randint(data_range[0], data_range[1])
        elif data_type == float:
            return random.uniform(data_range[0], data_range[1])
        elif data_type == str:
            length = data_range
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        elif data_type == bool:
            return random.choice([True, False])
        elif data_type == list:
            length = data_range['length']
            element_type = data_range['type']
            element_range = data_range.get('range', None)
            return [self.random_value(element_type, element_range) for _ in range(length)]
        elif data_type == tuple:
            length = data_range['length']
            element_type = data_range['type']
            element_range = data_range.get('range', None)
            return tuple(self.random_value(element_type, element_range) for _ in range(length))
        elif data_type == dict:
            return self.generate_structure(data_range)
        elif data_type == 'date':
            start_date = data_range[0]
            end_date = data_range[1]
            random_days = random.randint(0, (end_date - start_date).days)
            return start_date + timedelta(days=random_days)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def generate_structure(self, structure):
        """生成嵌套数据结构"""
        if not isinstance(structure, dict):
            raise ValueError("Structure must be a dictionary")
        
        result = {}
        for key, spec in structure.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Specification for '{key}' must be a dictionary")
            
            data_type = spec.get('type')
            if data_type is None:
                raise ValueError(f"Missing 'type' in specification for '{key}'")
            
            # 处理嵌套结构
            if data_type == dict:
                subs = spec.get('subs')
                if subs is None:
                    raise ValueError(f"Missing 'subs' for dict type in '{key}'")
                result[key] = self.generate_structure(subs)
            # 处理列表/元组嵌套
            elif data_type in (list, tuple):
                element_spec = spec.get('element_spec')
                if element_spec is None:
                    raise ValueError(f"Missing 'element_spec' for {data_type.__name__} in '{key}'")
                length = spec.get('length', 3)  # 默认长度3
                result[key] = self.random_value(
                    data_type, 
                    {'length': length, 'type': element_spec['type'], 'range': element_spec.get('range')}
                )
            # 处理基本类型
            else:
                result[key] = self.random_value(data_type, spec.get('range'))
        
        return result
    
    def generate_samples(self, num_samples, structure):
        """生成多个样本"""
        self.samples = [self.generate_structure(structure) for _ in range(num_samples)]
        return self.samples


def generate_structured_samples(**kwargs):
    """
    生成结构化随机样本的主函数
    参数:
        num_samples: 样本数量
        structure: 数据结构定义
    返回:
        生成的样本列表
    """
    if 'num_samples' not in kwargs:
        raise ValueError("Missing required parameter: num_samples")
    if 'structure' not in kwargs:
        raise ValueError("Missing required parameter: structure")
    
    sampler = DataSampler()
    return sampler.generate_samples(kwargs['num_samples'], kwargs['structure'])


# 示例用法
if __name__ == "__main__":
    # 定义复杂嵌套数据结构
    user_structure = {
        'id': {'type': int, 'range': (1000, 9999)},
        'username': {'type': str, 'range': 8},
        'email': {'type': str, 'range': 15},
        'is_active': {'type': bool},
        'created_at': {'type': 'date', 'range': [datetime(2020, 1, 1), datetime(2023, 12, 31)]},
        'profile': {
            'type': dict,
            'subs': {
                'full_name': {'type': str, 'range': 20},
                'age': {'type': int, 'range': (18, 65)},
                'height': {'type': float, 'range': (150.0, 200.0)},
                'address': {'type': str, 'range': 30}
            }
        },
        'preferences': {
            'type': dict,
            'subs': {
                'theme': {'type': str, 'range': 5},
                'notifications': {'type': bool},
                'language': {'type': str, 'range': 10}
            }
        },
        'scores': {
            'type': list,
            'element_spec': {'type': float, 'range': (0.0, 100.0)},
            'length': 5
        },
        'tags': {
            'type': tuple,
            'element_spec': {'type': str, 'range': 3},
            'length': 4
        },
        'connections': {
            'type': list,
            'element_spec': {
                'type': dict,
                'subs': {
                    'user_id': {'type': int, 'range': (1000, 9999)},
                    'connection_type': {'type': str, 'range': 10}
                }
            },
            'length': 3
        }
    }

    # 生成5个样本
    samples = generate_structured_samples(num_samples=5, structure=user_structure)
    
    # 打印生成的样本
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        for key, value in sample.items():
            print(f"  {key}: {value}")
