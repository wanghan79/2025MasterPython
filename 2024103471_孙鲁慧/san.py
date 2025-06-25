import random
import string
import math
import functools


def random_str(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_random_value(dtype):
    if dtype == 'int':
        return random.randint(0, 100)
    elif dtype == 'float':
        return round(random.uniform(0, 100), 2)
    elif dtype == 'str':
        return random_str()
    elif dtype == 'bool':
        return random.choice([True, False])
    else:
        raise ValueError(f"Unsupported type: {dtype}")


def generate_structure(template):
    if isinstance(template, dict):
        return {k: generate_structure(v) for k, v in template.items()}
    elif isinstance(template, list):
        return [generate_structure(v) for v in template]
    elif isinstance(template, tuple):
        return tuple(generate_structure(v) for v in template)
    elif isinstance(template, str):
        return generate_random_value(template)
    else:
        raise TypeError(f"Unsupported type: {type(template)}")


def get_all_numbers(data):
    nums = []
    if isinstance(data, (int, float)):
        nums.append(data)
    elif isinstance(data, dict):
        for v in data.values():
            nums.extend(get_all_numbers(v))
    elif isinstance(data, (list, tuple)):
        for item in data:
            nums.extend(get_all_numbers(item))
    return nums


# 带参数修饰器：支持 SUM, AVG, VAR, RMSE
def stats_decorator(*metrics):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            nums = []
            for sample in samples:
                nums.extend(get_all_numbers(sample))
            stats = {}
            n = len(nums)
            if 'SUM' in metrics:
                stats['SUM'] = sum(nums)
            if 'AVG' in metrics:
                stats['AVG'] = sum(nums) / n if n else 0
            if 'VAR' in metrics:
                mean = sum(nums) / n if n else 0
                stats['VAR'] = sum((x - mean) ** 2 for x in nums) / n if n else 0
            if 'RMSE' in metrics:
                stats['RMSE'] = math.sqrt(sum(x**2 for x in nums) / n) if n else 0
            return stats
        return wrapper
    return decorator


# 封装样本生成函数
@stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')  # 可改为任意组合
def generate_samples(sample_num=1, **kwargs):
    template = kwargs.get('structure')
    if not template:
        raise ValueError("Please provide a 'structure' argument in kwargs.")
    return [generate_structure(template) for _ in range(sample_num)]

# 示例调用


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