import random
import string
from datetime import datetime, timedelta
import numpy as np
from functools import wraps

def stats_decorator(*stats_to_calculate):
    """
    带参数的装饰器，用于计算数据样本的统计特征
    参数:
        stats_to_calculate: 需要计算的统计项 ('sum', 'mean', 'variance', 'rmse')
    """
    def decorator(func_to_decorate):
        @wraps(func_to_decorate)
        def wrapper(sampler_instance, *args, **kwargs):
            # 调用原始函数生成样本
            generated_samples = func_to_decorate(sampler_instance, *args, **kwargs)
            
            # 对每个样本进行统计分析
            all_stats = []
            for sample in generated_samples:
                # 获取样本的所有叶节点
                leaf_nodes = sampler_instance.get_leaf_nodes(sample)
                
                # 提取数值型叶节点
                numeric_leaf_nodes = [leaf for leaf in leaf_nodes if isinstance(leaf['value'], (int, float))]
                numeric_values = [leaf['value'] for leaf in numeric_leaf_nodes]
                
                # 计算统计量
                stats_result = {}
                if numeric_values:
                    if 'sum' in stats_to_calculate:
                        stats_result['sum'] = np.sum(numeric_values)
                    if 'mean' in stats_to_calculate:
                        stats_result['mean'] = np.mean(numeric_values)
                    if 'variance' in stats_to_calculate:
                        stats_result['variance'] = np.var(numeric_values, ddof=0)
                    if 'rmse' in stats_to_calculate:
                        stats_result['rmse'] = np.sqrt(np.mean(np.square(numeric_values)))
                
                all_stats.append(stats_result)
            
            return generated_samples, all_stats
        return wrapper
    return decorator

class DataSampler:
    def __init__(self, num_samples=3):
        self.num_samples = num_samples
        self.generated_samples = []
    
    def random_value(self, data_type, data_range=None):
        """生成指定类型的随机值"""
        if data_type == int:
            return random.randint(data_range[0], data_range[1])
        elif data_type == float:
            return random.uniform(data_range[0], data_range[1])
        elif data_type == str:
            return ''.join(random.choices(string.ascii_uppercase, k=data_range))
        elif data_type == bool:
            return random.choice([True, False])
        elif data_type == list:
            return [self.random_value(data_range['type'], data_range.get('range')) 
                    for _ in range(data_range['length'])]
        elif data_type == tuple:
            return tuple(self.random_value(data_range['type'], data_range.get('range')) 
                        for _ in range(data_range['length']))
        elif data_type == dict:
            return self.generate_structure(data_range)
        elif data_type == 'date':
            start_date = data_range[0]
            end_date = data_range[1]
            random_days = random.randint(0, (end_date - start_date).days)
            return start_date + timedelta(days=random_days)
        else:
            return None
    
    def generate_structure(self, structure):
        """生成嵌套数据结构"""
        if isinstance(structure, dict):
            node = {}
            for k, v_def in structure.items():
                if isinstance(v_def, dict):
                    data_type = v_def.get('type')
                    data_range_param = v_def.get('range')
                    subs_param = v_def.get('subs', [])
                    
                    if isinstance(subs_param, list) and subs_param:
                        node[k] = [self.generate_structure(sub_struct) 
                                  for sub_struct in subs_param]
                    else:
                        node[k] = self.random_value(data_type, data_range_param)
                else:
                    raise ValueError(f"Field definition for '{k}' must be a dictionary.")
            return node
        else:
            raise ValueError("Initial structure must be a dictionary.")
    
    @stats_decorator('sum', 'mean', 'variance', 'rmse')
    def generate_samples(self, structure):
        """生成样本集，并自动计算统计特征"""
        self.generated_samples = [self.generate_structure(structure) 
                                for _ in range(self.num_samples)]
        return self.generated_samples
    
    def get_leaf_nodes(self, data, current_path=""):
        """
        递归获取数据结构中的所有叶节点
        返回: [{'path': 节点路径, 'value': 节点值}]
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

# 示例用法
if __name__ == "__main__":
    # 定义数据结构
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
            'range': {'type': float, 'range': (0.0, 100.0), 'length': 5}
        },
        'tags': {
            'type': tuple,
            'range': {'type': str, 'range': 3, 'length': 4}
        }
    }

    # 创建数据采样器
    sampler = DataSampler(num_samples=3)
    
    # 生成样本并自动计算统计特征
    samples, stats_results = sampler.generate_samples(user_structure)
    
    print("生成的用户样本及统计分析:")
    for i, (sample, stats) in enumerate(zip(samples, stats_results), 1):
        print(f"\n=== 用户样本 {i} ===")
        print("数据结构:")
        for key, value in sample.items():
            print(f"  {key}: {value}")
        
        print("\n数值型叶节点统计:")
        if stats:
            for stat_name, stat_value in stats.items():
                print(f"  {stat_name.upper()}: {stat_value:.4f}")
        else:
            print("  没有数值型叶节点")
    
    # 测试不同统计项组合
    print("\n测试不同统计项组合:")
    
    # 只计算均值和方差
    class CustomSampler(DataSampler):
        @stats_decorator('mean', 'variance')
        def generate_samples(self, structure):
            return super().generate_samples(structure)
    
    custom_sampler = CustomSampler(num_samples=2)
    samples, stats_results = custom_sampler.generate_samples(user_structure)
    
    for i, (sample, stats) in enumerate(zip(samples, stats_results), 1):
        print(f"\n样本 {i} 统计 (仅均值和方差):")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name.upper()}: {stat_value:.4f}")
