import random
import string
import datetime
import math
from functools import wraps

# ========== 作业2：结构化数据生成器 ==========
def random_int():
    return random.randint(0, 100)

def random_float():
    return round(random.uniform(0, 100), 2)

def random_str(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_bool():
    return random.choice([True, False])

def random_date(start_year=2000, end_year=2025):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = (end - start).days
    return (start + datetime.timedelta(days=random.randint(0, delta))).isoformat()

TYPE_GENERATORS = {
    'int': random_int,
    'float': random_float,
    'str': random_str,
    'bool': random_bool,
    'date': random_date
}

def generate_sample(structure):
    if isinstance(structure, dict):
        return {k: generate_sample(v) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [generate_sample(structure[0]) for _ in range(random.randint(1, 3))]
    elif isinstance(structure, tuple):
        return tuple(generate_sample(item) for item in structure)
    elif isinstance(structure, str):
        if structure.startswith("str:"):
            length = int(structure.split(":")[1])
            return random_str(length)
        elif structure in TYPE_GENERATORS:
            return TYPE_GENERATORS[structure]()
        else:
            raise ValueError(f"Unsupported type: {structure}")
    else:
        raise TypeError(f"Invalid structure type: {type(structure)}")

def DataSampler(sample_count=1, **kwargs):
    return [generate_sample(kwargs) for _ in range(sample_count)]

# ========== 修饰器：统计分析功能 ==========

def stats_decorator(stats=('SUM', 'AVG', 'VAR', 'RMSE')):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            values = extract_numeric_values(data)
            results = {}
            if not values:
                return {stat: None for stat in stats}

            n = len(values)
            s = sum(values)
            mean = s / n

            if 'SUM' in stats:
                results['SUM'] = s
            if 'AVG' in stats:
                results['AVG'] = mean
            if 'VAR' in stats:
                results['VAR'] = sum((x - mean) ** 2 for x in values) / n
            if 'RMSE' in stats:
                results['RMSE'] = math.sqrt(sum(x ** 2 for x in values) / n)
            return results
        return wrapper
    return decorator

def extract_numeric_values(data):
    result = []
    if isinstance(data, dict):
        for v in data.values():
            result.extend(extract_numeric_values(v))
    elif isinstance(data, list) or isinstance(data, tuple):
        for item in data:
            result.extend(extract_numeric_values(item))
    elif isinstance(data, (int, float)):
        result.append(data)
    return result

# ========== 使用示例 ==========

structure = {
    'user': {
        'id': 'int',
        'age': 'int'
    },
    'scores': ['float'],
    'valid': 'bool'
}

@stats_decorator(stats=('SUM', 'AVG', 'VAR', 'RMSE'))
def generate_user_data():
    return DataSampler(sample_count=5, **structure)

if __name__ == "__main__":
    result = generate_user_data()
    print("统计结果：")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")
