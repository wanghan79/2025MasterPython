import random
import string
import datetime
from typing import Any, Union

def generate_value(dtype: str, **kwargs):

    if dtype == "int":
        return random.randint(kwargs.get("int_min", 0), kwargs.get("int_max", 100))
    elif dtype == "float":
        return round(random.uniform(kwargs.get("float_min", 0), kwargs.get("float_max", 1)), 4)
    elif dtype == "str":
        length = kwargs.get("str_length", 8)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    elif dtype == "bool":
        return random.choice([True, False])
    elif dtype == "date":
        start = kwargs.get("date_start", datetime.date(2000, 1, 1))
        end = kwargs.get("date_end", datetime.date(2025, 12, 31))
        delta = end - start
        random_days = random.randint(0, delta.days)
        return start + datetime.timedelta(days=random_days)
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

def data_sampler(structure: Any, num: int = 1, **kwargs) -> list:

    def generate_structure(s):
        if isinstance(s, dict):
            return {k: generate_structure(v) for k, v in s.items()}
        elif isinstance(s, list):
            return [generate_structure(v) for v in s]
        elif isinstance(s, tuple):
            return tuple(generate_structure(v) for v in s)
        elif isinstance(s, str):
            return generate_value(s, **kwargs)
        else:
            raise ValueError(f"Unsupported structure type: {type(s)}")

    return [generate_structure(structure) for _ in range(num)]


# 示例使用
if __name__ == "__main__":

    structure = {
        "user_id": "int",
        "username": "str",
        "score": "float",
        "active": "bool",
        "created_at": "date",
        "profile": {
            "email": "str",
            "tags": ["str", "str"],
            "position": ("float", "float")
        }
    }


    samples = data_sampler(
        structure,
        num=5,
        str_length=6,
        int_min=1000,
        int_max=9999,
        float_min=0.0,
        float_max=100.0,
        date_start=datetime.date(2010, 1, 1),
        date_end=datetime.date(2025, 12, 31)
    )

    for idx, sample in enumerate(samples, 1):
        print(f"Sample {idx}:")
        print(sample)
        print()
