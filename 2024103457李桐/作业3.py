import random
import string
import math
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Tuple, Union, Callable
import random
import string
import math
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Tuple, Union, Callable

def DataSampler(**kwargs) -> List[Dict[str, Any]]:
    num_samples = kwargs.get('num_samples', 10)
    nested_structure = kwargs.get('nested_structure')
    
    if nested_structure is None:
        raise ValueError("nested_structure is required")
    
    samples = []
    for _ in range(num_samples):
        samples.append(_generate_data(nested_structure))
    return samples

def _generate_data(structure):
    if isinstance(structure, dict):
        return {k: _generate_data(v) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [_generate_data(item) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(_generate_data(item) for item in structure)
    elif structure == 'int':
        return random.randint(1, 100)
    elif structure == 'float':
        return round(random.uniform(1.0, 100.0), 2)
    elif structure == 'str':
        length = random.randint(5, 20)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    elif structure == 'bool':
        return random.choice([True, False])
    elif structure == 'date':
        start = datetime(2000, 1, 1)
        end = datetime(2023, 12, 31)
        delta = end - start
        random_days = random.randint(0, delta.days)
        return (start + timedelta(days=random_days)).date()
    else:
        raise ValueError(f"Unsupported type: {structure}")

def stats_decorator(*stats_args):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            analyzer = DataAnalyzer(samples)
            return analyzer.analyze(stats_args)
        return wrapper
    return decorator

class DataAnalyzer:
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
        self.numeric_values = self._collect_numeric_values()

    def _collect_numeric_values(self) -> List[Union[int, float]]:
        numeric_values = []
        for sample in self.samples:
            self._traverse(sample, numeric_values)
        return numeric_values

    def _traverse(self, data, numeric_values):
        if isinstance(data, dict):
            for v in data.values():
                self._traverse(v, numeric_values)
        elif isinstance(data, (list, tuple)):
            for item in data:
                self._traverse(item, numeric_values)
        elif isinstance(data, (int, float)):
            numeric_values.append(data)

    def analyze(self, stats_args):
        results = {}
        if 'SUM' in stats_args:
            results['SUM'] = sum(self.numeric_values)
        if 'AVG' in stats_args:
            results['AVG'] = sum(self.numeric_values) / len(self.numeric_values) if self.numeric_values else 0
        if 'VAR' in stats_args:
            avg = results.get('AVG', sum(self.numeric_values) / len(self.numeric_values)) if self.numeric_values else 0
            results['VAR'] = sum((x - avg) ** 2 for x in self.numeric_values) / len(self.numeric_values) if self.numeric_values else 0
        if 'RMSE' in stats_args:
            avg = results.get('AVG', sum(self.numeric_values) / len(self.numeric_values)) if self.numeric_values else 0
            results['RMSE'] = math.sqrt(sum((x - avg) ** 2 for x in self.numeric_values) / len(self.numeric_values)) if self.numeric_values else 0
        return results

if __name__ == "__main__":
    sample_structure = {
        "id": "int",
        "name": "str",
        "active": "bool",
        "profile": {
            "age": "int",
            "score": "float"
        },
        "tags": ["str"],
        "metadata": ("int", "str", "bool")
    }

    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_samples(**kwargs):
        return DataSampler(**kwargs)

    result = generate_samples(num_samples=5, nested_structure=sample_structure)
    print("\nGenerated Samples with Statistics:")
    print(result)
def DataSampler(**kwargs) -> List[Dict[str, Any]]:
    num_samples = kwargs.get('num_samples', 10)
    nested_structure = kwargs.get('nested_structure')
    
    if nested_structure is None:
        raise ValueError("nested_structure is required")
    
    samples = []
    for _ in range(num_samples):
        samples.append(_generate_data(nested_structure))
    return samples

def _generate_data(structure):
    if isinstance(structure, dict):
        return {k: _generate_data(v) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [_generate_data(item) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(_generate_data(item) for item in structure)
    elif structure == 'int':
        return random.randint(1, 100)
    elif structure == 'float':
        return round(random.uniform(1.0, 100.0), 2)
    elif structure == 'str':
        length = random.randint(5, 20)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    elif structure == 'bool':
        return random.choice([True, False])
    elif structure == 'date':
        start = datetime(2000, 1, 1)
        end = datetime(2023, 12, 31)
        delta = end - start
        random_days = random.randint(0, delta.days)
        return (start + timedelta(days=random_days)).date()
    else:
        raise ValueError(f"Unsupported type: {structure}")

def stats_decorator(*stats_args):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            analyzer = DataAnalyzer(samples)
            return analyzer.analyze(stats_args)
        return wrapper
    return decorator

class DataAnalyzer:
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
        self.numeric_values = self._collect_numeric_values()

    def _collect_numeric_values(self) -> List[Union[int, float]]:
        numeric_values = []
        for sample in self.samples:
            self._traverse(sample, numeric_values)
        return numeric_values

    def _traverse(self, data, numeric_values):
        if isinstance(data, dict):
            for v in data.values():
                self._traverse(v, numeric_values)
        elif isinstance(data, (list, tuple)):
            for item in data:
                self._traverse(item, numeric_values)
        elif isinstance(data, (int, float)):
            numeric_values.append(data)

    def analyze(self, stats_args):
        results = {}
        if 'SUM' in stats_args:
            results['SUM'] = sum(self.numeric_values)
        if 'AVG' in stats_args:
            results['AVG'] = sum(self.numeric_values) / len(self.numeric_values) if self.numeric_values else 0
        if 'VAR' in stats_args:
            avg = results.get('AVG', sum(self.numeric_values) / len(self.numeric_values)) if self.numeric_values else 0
            results['VAR'] = sum((x - avg) ** 2 for x in self.numeric_values) / len(self.numeric_values) if self.numeric_values else 0
        if 'RMSE' in stats_args:
            avg = results.get('AVG', sum(self.numeric_values) / len(self.numeric_values)) if self.numeric_values else 0
            results['RMSE'] = math.sqrt(sum((x - avg) ** 2 for x in self.numeric_values) / len(self.numeric_values)) if self.numeric_values else 0
        return results

if __name__ == "__main__":
    sample_structure = {
        "id": "int",
        "name": "str",
        "active": "bool",
        "profile": {
            "age": "int",
            "score": "float"
        },
        "tags": ["str"],
        "metadata": ("int", "str", "bool")
    }

    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_samples(**kwargs):
        return DataSampler(**kwargs)

    result = generate_samples(num_samples=5, nested_structure=sample_structure)
    print("\nGenerated Samples with Statistics:")
    print(result)