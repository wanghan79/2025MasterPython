import random
import string
from datetime import datetime, timedelta

def generate_random_value(data_type, data_range=None):
    if data_type == int:
        return random.randint(*data_range)
    elif data_type == float:
        return round(random.uniform(*data_range), 2)
    elif data_type == str:
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=data_range))
    elif data_type == bool:
        return random.choice([True, False])
    elif data_type == 'timestamp':
        start, end = data_range
        delta = (end - start).total_seconds()
        return start + timedelta(seconds=random.randint(0, int(delta)))
    elif data_type == list:
        return [generate_random_value(data_range['type'], data_range['range']) for _ in range(data_range['length'])]
    elif data_type == tuple:
        return tuple(generate_random_value(data_range['type'], data_range['range']) for _ in range(data_range['length']))
    elif data_type == dict:
        return build_nested_structure(data_range)
    else:
        return None

def build_nested_structure(structure):
    data = {}
    for key, config in structure.items():
        if isinstance(config, dict) and 'type' in config:
            if config['type'] == dict and 'subs' in config:
                nested = {}
                for sub in config['subs']:
                    nested.update(build_nested_structure(sub))
                data[key] = nested
            else:
                data[key] = generate_random_value(config['type'], config.get('range'))
        else:
            raise ValueError(f"Invalid config for key: {key}")
    return data

def generate_samples(**kwargs):
    """
    Generate nested random data samples.

    Args:
        structure (dict): A nested structure definition.
        count (int): Number of samples to generate.

    Returns:
        list: A list of generated data samples.
    """
    structure = kwargs.get('structure')
    count = kwargs.get('count', 1)
    return [build_nested_structure(structure) for _ in range(count)]


# 示例结构：模拟设备数据结构（完全自定义，非姓名年龄类）
device_structure = {
    'device_id': {'type': str, 'range': 10},  # 10位随机字符
    'is_active': {'type': bool},
    'last_checked': {'type': 'timestamp', 'range': [datetime(2024, 1, 1), datetime(2025, 1, 1)]},
    'sensors': {
        'type': dict,
        'subs': [
            {'temperature': {'type': float, 'range': (15.0, 90.0)}},
            {'pressure': {'type': float, 'range': (1.0, 5.0)}},
            {'vibration_levels': {'type': list, 'range': {'type': float, 'range': (0.0, 1.0), 'length': 3}}},
            {'status_flags': {'type': tuple, 'range': {'type': bool, 'range': None, 'length': 4}}}
        ]
    },
    'config': {
        'type': dict,
        'subs': [
            {'firmware': {'type': str, 'range': 6}},
            {'retry_limits': {'type': int, 'range': (1, 10)}}
        ]
    }
}

# 生成样本并打印
samples = generate_samples(structure=device_structure, count=3)

for i, sample in enumerate(samples, 1):
    print(f"📦 Sample {i}:")
    for key, value in sample.items():
        print(f"  {key}: {value}")
    print()
