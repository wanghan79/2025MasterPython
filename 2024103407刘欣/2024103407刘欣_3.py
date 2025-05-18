import random
import string
from datetime import datetime, timedelta
import numpy as np
from functools import wraps


# 定义统计修饰器
def stats_on_sample_decorator(*stats_to_calculate):
    def decorator(func_to_decorate):
        @wraps(func_to_decorate)
        def wrapper(sampler_instance, data_sample, *args, **kwargs):
            processed_data = func_to_decorate(sampler_instance, data_sample, *args, **kwargs)
            leaf_nodes = sampler_instance.get_leaf_nodes(processed_data)
            numeric_leaf_nodes = [leaf for leaf in leaf_nodes if isinstance(leaf['value'], (int, float))]

            # 初始化统计结果字典
            stats_result = {}

            # 按指定的统计量进行计算
            numeric_values = [leaf['value'] for leaf in numeric_leaf_nodes]
            if numeric_values:
                for stat_name in stats_to_calculate:
                    if stat_name == 'mean':
                        stats_result[stat_name] = np.mean(numeric_values)
                    elif stat_name == 'variance':
                        stats_result[stat_name] = np.var(numeric_values, ddof=0)  # 总体方差
                    elif stat_name == 'rmse':
                        stats_result[stat_name] = np.sqrt(np.var(numeric_values, ddof=0))
                    elif stat_name == 'sum':
                        stats_result[stat_name] = np.sum(numeric_values)


            # 返回原始结果和统计结果
            return processed_data, stats_result
        return wrapper
    return decorator


class DataSampler:
    def __init__(self, num_samples=5):
        self.num_samples = num_samples

    def random_value(self, data_type, data_range=None):
        if data_type == int:
            return random.randint(data_range[0], data_range[1])
        elif data_type == float:
            return random.uniform(data_range[0], data_range[1])
        elif data_type == str:
            return ''.join(random.choices(string.ascii_letters, k=data_range))
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
        """
        递归获取数据结构中的所有叶节点，并返回包含叶节点信息的列表。
        """
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
    data_structure_def = {
        'name': {'type': str, 'range': 5},
        'age': {'type': int, 'range': (0, 100)},
        'height': {'type': float, 'range': (0.0, 200.0)},
        'favorites': {
            'type': dict,
            'subs': [
                {'colors': {'type': list, 'range': {'type': str, 'range': 5, 'length': 3}}},
                {'numbers': {'type': tuple, 'range': {'type': int, 'range': (0, 10), 'length': 3}}},
                {'shapes': {'type': list, 'range': {'type': str, 'range': 6, 'length': 2}}},
                {'scores': {'type': list, 'range': {'type': float, 'range': (0.0, 100.0), 'length': 4}}},
                {'tags': {'type': tuple, 'range': {'type': str, 'range': 4, 'length': 3}}}
            ]
        },
        'subject': {'type': str, 'range': 10},
        'state': {'type': bool},
        'date': {'type': 'date', 'range': [datetime(2020, 1, 1), datetime(2023, 12, 31)]}
    }

    sampler = DataSampler(num_samples=5)
    generated_samples_list = sampler.generate_samples(data_structure_def)

    print("生成的样本及各自的统计分析:")
    for i, sample_item in enumerate(generated_samples_list):
        print(f"\n--- 样本 {i+1} ---")
        analyzed_data, sample_stats = sampler.analyze_sample(sample_item)

        print("原始数据:")
        print(analyzed_data)
        print("该样本的统计结果 (针对所有数值型叶节点):")
        print(sample_stats)