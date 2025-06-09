import random
import string
from typing import Any

def random_value(value_type: type, int_range=(0, 100), float_range=(0.0, 1.0), str_length=8):
    if value_type == int:
        return random.randint(*int_range)
    elif value_type == float:
        return random.uniform(*float_range)
    elif value_type == str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=str_length))
    else:
        return None

def generate_sample(structure: Any, int_range, float_range, str_length):
    if isinstance(structure, dict):
        return {
            key: generate_sample(val, int_range, float_range, str_length)
            for key, val in structure.items()
        }
    elif isinstance(structure, list):
        return [
            generate_sample(val, int_range, float_range, str_length)
            for val in structure
        ]
    elif isinstance(structure, tuple):
        return tuple(
            generate_sample(val, int_range, float_range, str_length)
            for val in structure
        )
    elif structure in [int, float, str]:
        return random_value(structure, int_range, float_range, str_length)
    else:
        return None

def generate_nested_samples(**kwargs):
    structure = kwargs.get("structure")
    num_samples = kwargs.get("num_samples", 1)
    int_range = kwargs.get("int_range", (0, 100))
    float_range = kwargs.get("float_range", (0.0, 1.0))
    str_length = kwargs.get("str_length", 8)

    return [
        generate_sample(structure, int_range, float_range, str_length)
        for _ in range(num_samples)
    ]
