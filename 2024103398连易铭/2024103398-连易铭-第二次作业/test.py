import random
from random import randint, uniform, choice
from string import ascii_letters
from datetime import datetime as dt
from datetime import timedelta as td
from uuid import uuid4 as new_uuid

class MockDataGenerator:
    def __init__(self, sample_count=None):
        self.sample_count = sample_count if sample_count else randint(1, 100)
    
    def _create_random_value(self, value_type, constraints=None):
        """
        Generate random data based on specified type and constraints
        """
        if value_type == int:
            return randint(constraints[0], constraints[1])
        elif value_type == float:
            return round(uniform(*constraints), 2)
        elif value_type == str:
            return ''.join(choice(ascii_letters) for _ in range(constraints))
        elif value_type == bool:
            return choice((True, False))
        elif value_type == list:
            return [self._create_random_value(constraints['type'], constraints['range']) 
                   for _ in range(constraints['length'])]
        elif value_type == tuple:
            return tuple(self._create_random_value(constraints['type'], constraints['range']) 
                         for _ in range(constraints['length']))
        elif value_type == dict:
            return self._build_complex_structure(constraints)
        elif value_type == 'date':
            start, end = constraints
            delta = (end - start).days
            return start + td(days=randint(0, delta))
        elif value_type == 'uuid':
            return str(new_uuid())
        else:
            raise TypeError(f"Type not supported: {value_type}")

    def _build_complex_structure(self, structure_def):
        """
        Recursively build nested data structures
        """
        if not isinstance(structure_def, dict):
            raise ValueError("Structure definition must be a dictionary")
        
        result = {}
        for field, config in structure_def.items():
            if isinstance(config, dict):
                if 'subs' in config:
                    result[field] = [self._build_complex_structure(sub) 
                                    for sub in config['subs']]
                else:
                    val_type = config.get('type')
                    val_range = config.get('range')
                    result[field] = self._create_random_value(val_type, val_range)
            else:
                raise ValueError("Invalid configuration format")
        return result

    def generate_mock_data(self, **structure_template):
        """
        Produce multiple mock data samples based on template
        """
        return [self._build_complex_structure(structure_template) 
                for _ in range(self.sample_count)]


def create_mock_samples(**template):
    generator = MockDataGenerator()
    return generator.generate_mock_data(**template)


# Example usage with random sample count
if __name__ == '__main__':
    example_data = create_mock_samples(
        name={'type': str, 'range': 5},
        age={'type': int, 'range': (1, 99)},
        height={'type': float, 'range': (50.0, 250.0)},
        preferences={
            'type': dict,
            'subs': [
                {'colors': {'type': list, 'range': {'type': str, 'range': 5, 'length': 4}}},
                {'lucky_numbers': {'type': tuple, 'range': {'type': int, 'range': (1, 100), 'length': 5}}},
                {'interests': {'type': list, 'range': {'type': str, 'range': 8, 'length': 3}}},
                {'ratings': {'type': list, 'range': {'type': float, 'range': (0.0, 5.0), 'length': 10}}}
            ]
        },
        department={'type': str, 'range': 15},
        is_active={'type': bool},
        join_date={'type': 'date', 'range': [dt(2018, 1, 1), dt(2023, 12, 31)]},
        employee_id={'type': 'uuid'}
    )

    print(f"\nGenerated {len(example_data)} random samples:")
    for idx, sample in enumerate(example_data, start=1):
        print(f"\nSample #{idx}:")
        print(sample)
