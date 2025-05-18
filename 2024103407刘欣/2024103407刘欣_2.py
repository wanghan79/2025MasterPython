import random
import string
from datetime import datetime, timedelta

class DataSampler:
    def __init__(self, num_samples=5):
        self.num_samples = num_samples
        self.samples = []

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
            return [self.random_value(data_range['type'], data_range['range']) for _ in range(data_range['length'])]
        elif data_type == tuple:
            return tuple(self.random_value(data_range['type'], data_range['range']) for _ in range(data_range['length']))
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
            for k, v in structure.items():
                if isinstance(v, dict):
                    data_type = v.get('type')
                    data_range = v.get('range')
                    subs = v.get('subs', [])
                    if isinstance(subs, list) and subs:
                        node[k] = [self.generate_structure(sub) for sub in subs]
                    else:
                        node[k] = self.random_value(data_type, data_range)
                else:
                    raise ValueError("Expected dictionary structure")
            return node
        else:
            raise ValueError("Unsupported structure")

    def generate_samples(self, structure):
        self.samples = [self.generate_structure(structure) for _ in range(self.num_samples)]
        return self.samples

# Example usage
data_structure = {
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
samples = sampler.generate_samples(data_structure)
for sample in samples:
    print(sample)