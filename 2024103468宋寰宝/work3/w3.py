import random
import string
import math
import functools
from collections.abc import Mapping, Sequence


def random_str(length=6):
    return ''.join(
        random.choices(string.ascii_letters + string.digits, k=length))


def generate_random_value(dtype):
    generators = {
        'int': lambda: random.randint(0, 100),
        'float': lambda: round(random.uniform(0, 100), 2),
        'str': lambda: random_str(),
        'bool': lambda: random.choice([True, False])
    }
    try:
        return generators[dtype]()
    except KeyError:
        raise ValueError(f"Unsupported type: {dtype}")


def generate_structure(template):
    if isinstance(template, Mapping):
        return {k: generate_structure(v) for k, v in template.items()}
    if isinstance(template, Sequence) and not isinstance(template, str):
        return type(template)(generate_structure(v) for v in template)
    return generate_random_value(template)


def get_all_numbers(data):
    if isinstance(data, (int, float)):
        yield data
    elif isinstance(data, Mapping):
        yield from (num for v in data.values() for num in get_all_numbers(v))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        yield from (num for item in data for num in get_all_numbers(item))


def calculate_sum(nums):
    return sum(nums)


def calculate_avg(nums):
    nums = list(nums)
    n = len(nums)
    return sum(nums) / n if n else 0


def calculate_var(nums):
    nums = list(nums)
    n = len(nums)
    if n == 0:
        return 0
    mean = sum(nums) / n
    return sum((x - mean) ** 2 for x in nums) / n


def calculate_rmse(nums):
    nums = list(nums)
    n = len(nums)
    return math.sqrt(sum(x ** 2 for x in nums) / n) if n else 0


METRIC_FUNCTIONS = {
    'SUM': calculate_sum,
    'AVG': calculate_avg,
    'VAR': calculate_var,
    'RMSE': calculate_rmse
}


def stats_decorator(*metrics):
    invalid_metrics = [m for m in metrics if m not in METRIC_FUNCTIONS]
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {', '.join(invalid_metrics)}")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            nums = list(get_all_numbers(samples))

            if not nums and any(m in metrics for m in {'AVG', 'VAR', 'RMSE'}):
                return {metric: 0 for metric in metrics}

            return {
                metric: METRIC_FUNCTIONS[metric](nums)
                for metric in metrics
            }

        return wrapper

    return decorator


@stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def generate_samples(sample_num=1, **kwargs):
    template = kwargs.get('structure')
    if not template:
        raise ValueError("Please provide a 'structure' argument in kwargs.")
    return [generate_structure(template) for _ in range(sample_num)]


if __name__ == "__main__":
    struct_template = {
        'id': 'int',
        'score': 'float',
        'meta': {
            'a': 'int',
            'b': 'float'
        },
        'tuple_data': ('float', 'int'),
        'list_data': ['int', 'float'],
        'desc': 'str'
    }

    result = generate_samples(sample_num=10, structure=struct_template)
    print("统计结果：", result)