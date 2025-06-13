import numpy as np
import random as rnd
from string import ascii_letters as letters
from datetime import datetime as dt, timedelta as delta
from uuid import uuid4 as generate_id
from functools import wraps

class MockDataFactory:
    def __init__(self, count=1):
        self.count = count
    
    def _make_random(self, val_type, constraints=None):
        """Generate random data of specified type"""
        if val_type == int:
            return rnd.randint(*constraints)
        elif val_type == float:
            return round(rnd.uniform(*constraints), 2)
        elif val_type == str:
            return ''.join(rnd.choices(letters, k=constraints))
        elif val_type == bool:
            return rnd.choice([False, True])
        elif val_type == list:
            return [self._make_random(constraints['type'], constraints['range']) 
                   for _ in range(constraints['length'])]
        elif val_type == tuple:
            return tuple(self._make_random(constraints['type'], constraints['range']) 
                         for _ in range(constraints['length']))
        elif val_type == dict:
            return self._build_nested(constraints)
        elif val_type == 'date':
            days_diff = (constraints[1] - constraints[0]).days
            return constraints[0] + delta(days=rnd.randint(0, days_diff))
        elif val_type == 'uuid':
            return str(generate_id())
        else:
            raise TypeError(f"Invalid type specified: {val_type}")

    def _build_nested(self, template):
        """Recursively construct nested data structures"""
        if not isinstance(template, dict):
            raise ValueError("Template must be dictionary")
        
        result = {}
        for key, spec in template.items():
            if isinstance(spec, dict):
                if 'subs' in spec:
                    result[key] = [self._build_nested(sub) for sub in spec['subs']]
                else:
                    result[key] = self._make_random(spec.get('type'), spec.get('range'))
            else:
                raise ValueError("Invalid specification format")
        return result

    def produce_mock_data(self, **template):
        """Generate multiple mock data samples"""
        return [self._build_nested(template) for _ in range(self.count)]


def calculate_metrics(*metrics):
    """Decorator to compute statistical metrics on numeric data"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data_samples = func(*args, **kwargs)
            
            def extract_numbers(data, current_path=""):
                nums = []
                if isinstance(data, dict):
                    for k, v in data.items():
                        path = f"{current_path}.{k}" if current_path else k
                        nums.extend(extract_numbers(v, path))
                elif isinstance(data, (list, tuple)):
                    for i, elem in enumerate(data):
                        path = f"{current_path}[{i}]"
                        nums.extend(extract_numbers(elem, path))
                elif isinstance(data, (int, float)):
                    nums.append((current_path, data))
                return nums
            
            for sample in data_samples:
                numeric_data = extract_numbers(sample)
                metrics_result = {}
                
                if numeric_data:
                    values = [val for _, val in numeric_data]
                    
                    if 'sum' in metrics:
                        metrics_result['total'] = np.sum(values)
                    if 'avg' in metrics:
                        metrics_result['mean'] = np.mean(values)
                    if 'var' in metrics:
                        metrics_result['variance'] = np.var(values)
                    if 'rmse' in metrics:
                        metrics_result['root_mean_square'] = np.sqrt(np.mean(np.square(values)))
                
                sample['metrics'] = metrics_result
            
            return data_samples
        return wrapper
    return decorator


@calculate_metrics('sum', 'avg', 'var', 'rmse')
def create_test_data(**structure):
    """Generate test data with random sample size"""
    sample_count = rnd.randint(5, 100)  # 5-100 samples
    factory = MockDataFactory(sample_count)
    return factory.produce_mock_data(**structure)


# Demonstration
if __name__ == '__main__':
    test_samples = create_test_data(
        identifier={'type': str, 'range': 8},
        years={'type': int, 'range': (18, 90)},
        weight={'type': float, 'range': (40.0, 150.0)},
        attributes={
            'type': dict,
            'subs': [
                {'codes': {'type': list, 'range': {'type': int, 'range': (1000, 9999), 'length': 5}}},
                {'coordinates': {'type': tuple, 'range': {'type': float, 'range': (-90.0, 90.0), 'length': 2}}},
                {'flags': {'type': list, 'range': {'type': bool, 'length': 4}}},
                {'percentages': {'type': list, 'range': {'type': float, 'range': (0.0, 1.0), 'length': 3}}}
            ]
        },
        department={'type': str, 'range': 12},
        active={'type': bool},
        created={'type': 'date', 'range': [dt(2015, 1, 1), dt(2022, 12, 31)]},
        record_id={'type': 'uuid'}
    )

    print(f"\nProduced {len(test_samples)} test samples with metrics:")
    for idx, sample in enumerate(test_samples, 1):
        print(f"\nSample #{idx}:")
        # Display original data
        for field, value in sample.items():
            if field != 'metrics':
                print(f"{field:>12}: {value}")
        # Display calculated metrics
        print("     Metrics:", sample.get('metrics', {}))
