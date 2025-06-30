import random
import string
import math
from datetime import datetime, timedelta
from functools import wraps


def stats_decorator(stats):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            results = analyze(samples, stats)
            return samples, results

        return wrapper

    return decorator


def analyze(samples, stats):
    numbers = extract_numbers(samples)
    n = len(numbers)
    if n == 0:
        return {stat: None for stat in stats}

    total = sum(numbers)
    avg = total / n
    sum_sq = sum(x * x for x in numbers)

    results = {}
    if 'SUM' in stats: results['SUM'] = total
    if 'AVG' in stats: results['AVG'] = avg
    if 'VAR' in stats:
        results['VAR'] = (sum_sq - n * avg * avg) / (n - 1) if n > 1 else 0.0
    if 'RMSE' in stats:
        results['RMSE'] = math.sqrt(sum_sq / n)
    return results


def extract_numbers(data):
    numbers = []
    if isinstance(data, (int, float)):
        numbers.append(data)
    elif isinstance(data, dict):
        for value in data.values():
            numbers.extend(extract_numbers(value))
    elif isinstance(data, (list, tuple)):
        for item in data:
            numbers.extend(extract_numbers(item))
    return numbers


@stats_decorator(stats=['SUM', 'AVG', 'VAR', 'RMSE'])
def DataSampler(n, **kwargs):
    def generate_value(spec):
        if isinstance(spec, type):
            if spec == int: return random.randint(1, 10000)
            if spec == float: return round(random.uniform(1.0, 1000.0), 2)
            if spec == str: return ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            if spec == bool: return random.choice([True, False])
            if spec == datetime:
                start = datetime(2000, 1, 1)
                end = datetime.now()
                return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))
            raise ValueError(f"Unsupported type: {spec}")
        if isinstance(spec, list): return [generate_value(spec[0]) for _ in range(spec[1])]
        if isinstance(spec, tuple): return tuple(generate_value(item) for item in spec)
        if isinstance(spec, dict): return {k: generate_value(v) for k, v in spec.items()}
        raise ValueError(f"Invalid specification: {spec}")

    samples = []
    for _ in range(n):
        sample = {}
        for key, spec in kwargs.items():
            sample[key] = generate_value(spec)
        samples.append(sample)
    return samples


if __name__ == "__main__":
    data_spec = {
        "id": int,
        "score": float,
        "active": bool,
        "scores": [float, 5],
        "details": {
            "rating": float,
            "counts": [int, 3],
            "metrics": (float, float, float)
        }
    }

    samples, stats = DataSampler(1000, **data_spec)

    print("\nGenerated Samples (first 3):")
    for i, sample in enumerate(samples[:3]):
        print(f"Sample {i + 1}: {sample}")

    print("\nStatistical Analysis:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.4f}" if isinstance(value, float) else f"{stat}: {value}")
