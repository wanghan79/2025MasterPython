import random
import string
from datetime import datetime, timedelta
import numpy as np
from functools import wraps

def stats_on_sample_decorator(*stats_to_calculate):
    def decorator(func_to_decorate):
        @wraps(func_to_decorate)
        def wrapper(sampler_instance, data_sample, *args, **kwargs):
            processed_data = func_to_decorate(sampler_instance, data_sample, *args, **kwargs)
            leaf_nodes = sampler_instance.get_leaf_nodes(processed_data)
            numeric_leaf_nodes = [leaf for leaf in leaf_nodes if isinstance(leaf['value'], (int, float))]
            stats_result = {}
            numeric_values = [leaf['value'] for leaf in numeric_leaf_nodes]
            if numeric_values:
                for stat_name in stats_to_calculate:
                    if stat_name == 'mean':
                        stats_result[stat_name] = np.mean(numeric_values)
                    elif stat_name == 'variance':
                        stats_result[stat_name] = np.var(numeric_values, ddof=0)
                    elif stat_name == 'rmse':
                        stats_result[stat_name] = np.sqrt(np.var(numeric_values, ddof=0))
                    elif stat_name == 'sum':
                        stats_result[stat_name] = np.sum(numeric_values)
            return processed_data, stats_result
        return wrapper
    return decorator

class DataSampler:
    def __init__(self, num_samples=3):
        self.num_samples = num_samples

    def random_value(self, data_type, data_range=None):
        if data_type == int:
            return random.randint(data_range[0], data_range[1])
        elif data_type == float:
            return random.uniform(data_range[0], data_range[1])
        elif data_type == str:
            return ''.join(random.choices(string.ascii_uppercase, k=data_range))
        elif data_type == bool:
            return random.choice([True, False])
        elif data_type == list:
            return [self.random_value(data_range['type'], data_range.get('range')) for _ in range(data_range['length'])]
        elif data_type == tuple:
            return tuple(self.random_value(data_range['type'], data_range.get('range')) for _ in range(data_range['length']))
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
        if isinstance(structure, dict):
            node = {}
            for k, v_def in structure.items():
                if isinstance(v_def, dict):
                    data_type = v_def.get('type')
                    data_range_param = v_def.get('range')
                    subs_param = v_def.get('subs', [])
                    if isinstance(subs_param, list) and subs_param:
                        node[k] = [self.generate_structure(sub_struct) for sub_struct in subs_param]
                    else:
                        node[k] = self.random_value(data_type, data_range_param)
                else:
                    raise ValueError(f"Field definition for '{k}' must be a dictionary.")
            return node
        else:
            raise ValueError("Initial structure to generate_structure must be a dictionary.")

    def generate_samples(self, structure):
        self.generated_samples = [self.generate_structure(structure) for _ in range(self.num_samples)]
        return self.generated_samples

    def get_leaf_nodes(self, data, current_path=""):
        leaf_nodes = []
        if isinstance(data, dict):
            for k, v_item in data.items():
                new_path = f"{current_path}.{k}" if current_path else k
                leaf_nodes.extend(self.get_leaf_nodes(v_item, new_path))
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                new_path = f"{current_path}[{i}]"
                leaf_nodes.extend(self.get_leaf_nodes(item, new_path))
        else:
            leaf_nodes.append({"path": current_path, "value": data})
        return leaf_nodes

    @stats_on_sample_decorator('mean', 'variance', 'rmse', 'sum')
    def analyze_sample(self, sample_data):
        return sample_data

if __name__ == "__main__":
    # 新的数据结构和变量名
    product_structure = {
        'product_name': {'type': str, 'range': 8},
        'quantity': {'type': int, 'range': (1, 50)},
        'weight': {'type': float, 'range': (0.1, 10.0)},
        'attributes': {
            'type': dict,
            'subs': [
                {'colors': {'type': list, 'range': {'type': str, 'range': 4, 'length': 2}}},
                {'sizes': {'type': tuple, 'range': {'type': int, 'range': (30, 45), 'length': 3}}},
                {'ratings': {'type': list, 'range': {'type': float, 'range': (1.0, 5.0), 'length': 5}}},
                {'labels': {'type': tuple, 'range': {'type': str, 'range': 3, 'length': 2}}}
            ]
        },
        'category': {'type': str, 'range': 6},
        'in_stock': {'type': bool},
        'added_date': {'type': 'date', 'range': [datetime(2022, 1, 1), datetime(2024, 5, 20)]}
    }

    sampler = DataSampler(num_samples=3)
    generated_products = sampler.generate_samples(product_structure)

    print("生成的产品样本及各自的统计分析:")
    for i, product in enumerate(generated_products):
        print(f"\n--- 产品样本 {i+1} ---")
        analyzed_data, sample_stats = sampler.analyze_sample(product)
        print("原始数据:")
        print(analyzed_data)
        print("该样本的统计结果 (针对所有数值型叶节点):")
        print(sample_stats)