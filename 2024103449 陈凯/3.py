import random
import string
from datetime import datetime, timedelta
import numpy as np
from functools import wraps

def compute_stats_decorator(*metrics):
    def decorator(func):
        @wraps(func)
        def wrapper(instance, sample, *args, **kwargs):
            result_data = func(instance, sample, *args, **kwargs)
            leaves = instance.extract_leaf_nodes(result_data)
            numeric_leaves = [leaf for leaf in leaves if isinstance(leaf['value'], (int, float))]
            stats = {}
            values = [leaf['value'] for leaf in numeric_leaves]
            if values:
                for metric in metrics:
                    if metric == 'mean':
                        stats[metric] = np.mean(values)
                    elif metric == 'variance':
                        stats[metric] = np.var(values, ddof=0)
                    elif metric == 'rmse':
                        stats[metric] = np.sqrt(np.var(values, ddof=0))
                    elif metric == 'sum':
                        stats[metric] = np.sum(values)
            return result_data, stats
        return wrapper
    return decorator

class DataSampler:
    def __init__(self, sample_count=3):
        self.sample_count = sample_count

    def random_value(self, val_type, val_range=None):
        if val_type == int:
            return random.randint(val_range[0], val_range[1])
        elif val_type == float:
            return random.uniform(val_range[0], val_range[1])
        elif val_type == str:
            return ''.join(random.choices(string.ascii_uppercase, k=val_range))
        elif val_type == bool:
            return random.choice([True, False])
        elif val_type == list:
            return [self.random_value(val_range['type'], val_range.get('range')) for _ in range(val_range['length'])]
        elif val_type == tuple:
            return tuple(self.random_value(val_range['type'], val_range.get('range')) for _ in range(val_range['length']))
        elif val_type == dict:
            return self.build_structure(val_range)
        elif val_type == 'date':
            start_date, end_date = val_range
            days_delta = (end_date - start_date).days
            random_offset = random.randint(0, days_delta)
            return start_date + timedelta(days=random_offset)
        else:
            return None

    def build_structure(self, template):
        if not isinstance(template, dict):
            raise ValueError("Template must be a dict")
        node = {}
        for key, val_def in template.items():
            if not isinstance(val_def, dict):
                raise ValueError(f"Definition of '{key}' must be a dict")
            data_type = val_def.get('type')
            data_range = val_def.get('range')
            substructures = val_def.get('subs', [])
            if isinstance(substructures, list) and substructures:
                node[key] = [self.build_structure(sub) for sub in substructures]
            else:
                node[key] = self.random_value(data_type, data_range)
        return node

    def generate_samples(self, template):
        self.samples = [self.build_structure(template) for _ in range(self.sample_count)]
        return self.samples

    def extract_leaf_nodes(self, data, path=""):
        leaves = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_path = f"{path}.{k}" if path else k
                leaves.extend(self.extract_leaf_nodes(v, new_path))
        elif isinstance(data, (list, tuple)):
            for i, elem in enumerate(data):
                new_path = f"{path}[{i}]"
                leaves.extend(self.extract_leaf_nodes(elem, new_path))
        else:
            leaves.append({"path": path, "value": data})
        return leaves

    @compute_stats_decorator('mean', 'variance', 'rmse', 'sum')
    def analyze_sample(self, sample):
        return sample


if __name__ == "__main__":
    product_template = {
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

    sampler = DataSampler(sample_count=3)
    products = sampler.generate_samples(product_template)

    print("生成的产品样本及统计信息:")
    for idx, product in enumerate(products, 1):
        print(f"\n--- 产品样本 {idx} ---")
        data, stats = sampler.analyze_sample(product)
        print("样本数据:")
        print(data)
        print("统计指标 (数值叶节点):")
        print(stats)
