import random
import string
import numpy as np
from datetime import datetime, timedelta
from functools import wraps

# 定义统计装饰器
def stats_decorator(*stats):
    """
    统计装饰器，用于计算数据样本中数值型叶节点的统计指标
    
    Args:
        *stats: 要计算的统计指标，可选值：'mean', 'variance', 'rmse', 'sum'
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原始方法
            result = func(*args, **kwargs)
            
            # 获取所有叶节点并筛选出数值型（int或float）的数据
            self = args[0]  # 获取实例对象
            leaf_nodes = self.get_leaf_nodes(result)
            numeric_leaf_nodes = [leaf for leaf in leaf_nodes if isinstance(leaf['value'], (int, float))]
            
            # 初始化统计结果字典
            stats_result = {}
            
            # 计算指定的统计指标
            numeric_values = [leaf['value'] for leaf in numeric_leaf_nodes]
            if numeric_values:  # 只有存在数值型数据时才计算统计量
                for stat in stats:
                    if stat == 'mean':
                        stats_result[stat] = np.mean(numeric_values)
                    elif stat == 'variance':
                        stats_result[stat] = np.var(numeric_values, ddof=0)  # 总体方差
                    elif stat == 'rmse':
                        stats_result[stat] = np.sqrt(np.var(numeric_values, ddof=0))
                    elif stat == 'sum':
                        stats_result[stat] = np.sum(numeric_values)
            
            # 返回原始结果和统计结果
            return result, stats_result
        
        return wrapper
    return decorator


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
    
    def get_leaf_nodes(self, data, current_path=""):
        """
        递归获取数据结构中的所有叶节点，并返回包含叶节点信息的列表
        
        Args:
            data: 数据结构（可以是dict、list、tuple或基本类型）
            current_path: 当前节点的路径
            
        Returns:
            包含叶节点信息的列表，每个叶节点信息是一个字典，包含'path'和'value'
        """
        leaf_nodes = []
        
        if isinstance(data, dict):
            for k, v in data.items():
                new_path = f"{current_path}.{k}" if current_path else k
                leaf_nodes.extend(self.get_leaf_nodes(v, new_path))
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                new_path = f"{current_path}[{i}]"
                leaf_nodes.extend(self.get_leaf_nodes(item, new_path))
        else:
            leaf_nodes.append({"path": current_path, "value": data})
            
        return leaf_nodes
    
    @stats_decorator('mean', 'variance', 'rmse', 'sum')
    def analyze(self, data):
        """
        分析数据样本，计算统计指标
        
        Args:
            data: 要分析的数据样本
            
        Returns:
            原始数据和统计结果
        """
        return data


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
    sample = sampler.generate_structure(data_structure)
    
    # 打印生成的样本
    print("生成的数据样本:")
    print(sample)
    print()
    
    # 分析样本并打印统计结果
    _, stats = sampler.analyze(sample)
    print("统计结果:")
    print(stats)