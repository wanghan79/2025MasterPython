import random
import string
from datetime import datetime, timedelta

def DataSampler(**kwargs):
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
        return random.randint(-1000, 1000)
    elif structure == 'float':
        return round(random.uniform(-1000.0, 1000.0), 2)
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
    
    samples = DataSampler(num_samples=3, nested_structure=sample_structure)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        for k, v in sample.items():
            print(f"{k}: {v}")