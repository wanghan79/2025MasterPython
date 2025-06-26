
import random
import string
from datetime import datetime, timedelta
import numpy as np
from functools import wraps

def sample_stats_calculator(*stats_to_compute):
    def decorator(decorated_function):
        @wraps(decorated_function)
        def wrapper(data_sampler, sample_item, *args, **kwargs):
            processed_sample = decorated_function(data_sampler, sample_item, *args, **kwargs)
            all_leaf_nodes = data_sampler.get_leaf_nodes(processed_sample)
            numeric_leaves = [leaf for leaf in all_leaf_nodes if isinstance(leaf['value'], (int, float))]
            calculated_stats = {}
            numeric_data = [leaf['value'] for leaf in numeric_leaves]
            if numeric_data:
                for stat_key in stats_to_compute:
                    if stat_key == 'mean':
                        calculated_stats[stat_key] = np.mean(numeric_data)
                    elif stat_key == 'variance':
                        calculated_stats[stat_key] = np.var(numeric_data, ddof=0)
                    elif stat_key == 'rmse':
                        calculated_stats[stat_key] = np.sqrt(np.var(numeric_data, ddof=0))
                    elif stat_key == 'sum':
                        calculated_stats[stat_key] = np.sum(numeric_data)
            return processed_sample, calculated_stats
        return wrapper
    return decorator

class DataSampler:
    def __init__(self, sample_count=3):
        self.sample_count = sample_count

    def get_random_value(self, value_type, value_range=None):
        if value_type == int:
            return random.randint(value_range[0], value_range[1])
        elif value_type == float:
            return random.uniform(value_range[0], value_range[1])
        elif value_type == str:
            return ''.join(random.choices(string.ascii_uppercase, k=value_range))
        elif value_type == bool:
            return random.choice([True, False])
        elif value_type == list:
            return [self.get_random_value(value_range['type'], value_range.get('range')) for _ in range(value_range['length'])]
        elif value_type == tuple:
            return tuple(self.get_random_value(value_range['type'], value_range.get('range')) for _ in range(value_range['length']))
        elif value_type == dict:
            return self.build_structure(value_range)
        elif value_type == 'date':
            date_start = value_range[0]
            date_end = value_range[1]
            days_offset = random.randint(0, (date_end - date_start).days)
            return date_start + timedelta(days=days_offset)
        else:
            return None

    def build_structure(self, schema_definition):
        if isinstance(schema_definition, dict):
            node_data = {}
            for key, field_definition in schema_definition.items():
                if isinstance(field_definition, dict):
                    field_type = field_definition.get('type')
                    field_range = field_definition.get('range')
                    sub_structures = field_definition.get('subs', [])
                    if isinstance(sub_structures, list) and sub_structures:
                        node_data[key] = [self.build_structure(sub_struct) for sub_struct in sub_structures]
                    else:
                        node_data[key] = self.get_random_value(field_type, field_range)
                else:
                    raise ValueError(f"Field definition for '{key}' must be a dictionary.")
            return node_data
        else:
            raise ValueError("Initial structure to build_structure must be a dictionary.")

    def generate_data_samples(self, schema_definition):
        self.output_samples = [self.build_structure(schema_definition) for _ in range(self.sample_count)]
        return self.output_samples

    def get_leaf_nodes(self, data_item, current_node_path=""):
        all_leaf_nodes = []
        if isinstance(data_item, dict):
            for key, node_value in data_item.items():
                next_path = f"{current_node_path}.{key}" if current_node_path else key
                all_leaf_nodes.extend(self.get_leaf_nodes(node_value, next_path))
        elif isinstance(data_item, (list, tuple)):
            for index, list_item in enumerate(data_item):
                next_path = f"{current_node_path}[{index}]"
                all_leaf_nodes.extend(self.get_leaf_nodes(list_item, next_path))
        else:
            all_leaf_nodes.append({"path": current_node_path, "value": data_item})
        return all_leaf_nodes

    @sample_stats_calculator('mean', 'variance', 'rmse', 'sum')
    def process_sample_for_stats(self, input_sample):
        return input_sample

if __name__ == "__main__":
    product_schema = {
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

    data_generator = DataSampler(sample_count=3)
    generated_items = data_generator.generate_data_samples(product_schema)

    for i, item in enumerate(generated_items):
        print(f"\nSample {i+1}:")
        item_data, item_stats = data_generator.process_sample_for_stats(item)
        print("Data:")
        print(item_data)
        print("Stats:")
        print(item_stats)
