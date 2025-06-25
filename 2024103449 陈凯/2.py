import random
import string
from datetime import datetime, timedelta

class DataSampler:
    def __init__(self, sample_count=5):
        self.sample_count = sample_count
        self.generated_samples = []

    def generate_random_value(self, value_type, value_range=None):
        if value_type == int:
            return random.randint(value_range[0], value_range[1])
        elif value_type == float:
            return random.uniform(value_range[0], value_range[1])
        elif value_type == str:
            return ''.join(random.choices(string.ascii_letters, k=value_range))
        elif value_type == bool:
            return random.choice([True, False])
        elif value_type == list:
            return [self.generate_random_value(value_range['type'], value_range['range']) for _ in range(value_range['length'])]
        elif value_type == tuple:
            return tuple(self.generate_random_value(value_range['type'], value_range['range']) for _ in range(value_range['length']))
        elif value_type == dict:
            return self.build_data_structure(value_range)
        elif value_type == 'date':
            start_date = value_range[0]
            end_date = value_range[1]
            days_span = (end_date - start_date).days
            random_offset = random.randint(0, days_span)
            return start_date + timedelta(days=random_offset)
        else:
            return None

    def build_data_structure(self, structure_def):
        if not isinstance(structure_def, dict):
            raise ValueError("Structure definition must be a dictionary")
        result_node = {}
        for field_name, field_def in structure_def.items():
            if not isinstance(field_def, dict):
                raise ValueError(f"Field definition for '{field_name}' must be a dictionary")
            field_type = field_def.get('type')
            field_range = field_def.get('range')
            field_subs = field_def.get('subs', [])
            if isinstance(field_subs, list) and field_subs:
                result_node[field_name] = [self.build_data_structure(sub_struct) for sub_struct in field_subs]
            else:
                result_node[field_name] = self.generate_random_value(field_type, field_range)
        return result_node

    def generate_samples(self, structure_definition):
        self.generated_samples = [self.build_data_structure(structure_definition) for _ in range(self.sample_count)]
        return self.generated_samples


# 使用示例
data_template = {
    'name': {'type': str, 'range': 8},  # 长度8字符串
    'age': {'type': int, 'range': (18, 60)},  # 18至60岁整数
    'height': {'type': float, 'range': (150.0, 200.0)},  # 150到200浮点数
    'favorites': {
        'type': dict,
        'subs': [
            {'colors': {'type': list, 'range': {'type': str, 'range': 3, 'length': 2}}},  # 两个长度为3的字符串
            {'numbers': {'type': tuple, 'range': {'type': int, 'range': (100, 200), 'length': 4}}},  # 四个100-200整数
            {'shapes': {'type': list, 'range': {'type': str, 'range': 4, 'length': 3}}},  # 三个长度为4的字符串
            {'scores': {'type': list, 'range': {'type': float, 'range': (60.0, 100.0), 'length': 5}}},  # 五个60~100浮点数
            {'tags': {'type': tuple, 'range': {'type': str, 'range': 2, 'length': 2}}}  # 两个长度为2的字符串
        ]
    },
    'subject': {'type': str, 'range': 6},  # 长度6字符串
    'state': {'type': bool},
    'date': {'type': 'date', 'range': [datetime(2022, 1, 1), datetime(2024, 12, 31)]}  # 2022年到2024年日期
}

sampler = DataSampler(sample_count=5)
samples = sampler.generate_samples(data_template)

for record in samples:
    print(record)
