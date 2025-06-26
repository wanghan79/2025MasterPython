import random
import string
import datetime
import math
from typing import Any, Union, Callable



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



def stats_decorator(*metrics: str) -> Callable:

    def decorator(func):
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            return {
                "data": data,
                "stats": analyze(data, metrics)
            }
        return wrapper
    return decorator


def analyze(data: list, metrics: tuple) -> dict:

    values = []

    def extract_values(item):
        if isinstance(item, dict):
            for v in item.values():
                extract_values(v)
        elif isinstance(item, list) or isinstance(item, tuple):
            for v in item:
                extract_values(v)
        elif isinstance(item, (int, float)):
            values.append(item)

    for entry in data:
        extract_values(entry)

    result = {}
    if not values:
        return {m: None for m in metrics}

    if "SUM" in metrics:
        result["SUM"] = sum(values)
    if "AVG" in metrics:
        result["AVG"] = sum(values) / len(values)
    if "VAR" in metrics:
        mean = sum(values) / len(values)
        result["VAR"] = sum((x - mean) ** 2 for x in values) / len(values)
    if "RMSE" in metrics:
        result["RMSE"] = math.sqrt(sum(x ** 2 for x in values) / len(values))

    return result



@stats_decorator("SUM", "AVG", "VAR", "RMSE")
def generate_user_data(num=5):
    structure = {
        "user_id": "int",
        "score": "float",
        "profile": {
            "likes": "int",
            "rating": "float",
            "tags": ["str", "str"]
        }
    }
    return data_sampler(structure, num=num, int_min=10, int_max=100, float_min=0, float_max=10)


if __name__ == "__main__":
    result = generate_user_data(num=5)

    print("=== 随机样本数据 ===")
    for i, item in enumerate(result["data"], 1):
        print(f"Sample {i}:", item)

    print("\n=== 数值统计结果 ===")
    for k, v in result["stats"].items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: 无")
