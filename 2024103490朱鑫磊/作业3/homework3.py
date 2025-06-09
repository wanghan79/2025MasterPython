import random
import string
import math
import functools


def generate_random_value(dtype):
    if dtype == int:
        return random.randint(0, 100)
    elif dtype == float:
        return round(random.uniform(0, 100), 2)
    elif dtype == str:
        return ''.join(random.choices(string.ascii_letters, k=5))
    elif dtype == bool:
        return random.choice([True, False])
    elif dtype == None:
        return None
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


def generate_sample(structure):
    if isinstance(structure, list):
        return [generate_sample(sub) for sub in structure]
    elif isinstance(structure, tuple):
        return tuple(generate_sample(sub) for sub in structure)
    elif isinstance(structure, set):
        return set(generate_sample(sub) for sub in structure)
    elif isinstance(structure, dict):
        return {k: generate_sample(v) for k, v in structure.items()}
    elif isinstance(structure, type):
        return generate_random_value(structure)
    else:
        raise TypeError(f"Invalid structure type: {type(structure)}")

def generate_samples(**kwargs):
    structure = kwargs.get("structure")
    num_samples = kwargs.get("num_samples", 1)
    if structure is None:
        raise ValueError("You must provide a 'structure' keyword argument.")
    return [generate_sample(structure) for _ in range(num_samples)]


def extract_numbers(data):
    numbers = []
    if isinstance(data, (int, float)):
        numbers.append(data)
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            numbers.extend(extract_numbers(item))
    elif isinstance(data, dict):
        for value in data.values():
            numbers.extend(extract_numbers(value))
    return numbers


def stat_decorator(*stats):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            values = extract_numbers(samples)

            results = {}
            n = len(values)
            if n == 0:
                return {stat: None for stat in stats}

            s = sum(values)
            avg = s / n
            var = sum((x - avg) ** 2 for x in values) / n
            rmse = math.sqrt(sum(x**2 for x in values) / n)

            if 'SUM' in stats:
                results['SUM'] = s
            if 'AVG' in stats:
                results['AVG'] = avg
            if 'VAR' in stats:
                results['VAR'] = var
            if 'RMSE' in stats:
                results['RMSE'] = rmse
            return results
        return wrapper
    return decorator


@stat_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def decorated_sample_generator(**kwargs):
    return generate_samples(**kwargs)

if __name__ == "__main__":
    structure_template = {
        "id": int,
        "score": float,
        "profile": {
            "math": float,
            "english": float,
            "age": int
        },
        "flag": bool,
        "tags": [int, float]
    }

    print("生成样本并统计数值型字段...")
    stats_result = decorated_sample_generator(structure=structure_template, num_samples=100)
    print("统计结果:")
    for k, v in stats_result.items():
        print(f"{k}: {v:.4f}")
