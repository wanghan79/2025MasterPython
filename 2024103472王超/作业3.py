import random
import string
import datetime
import math
from functools import wraps

# ---------- 作业二：样本生成 ----------
def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_date(start_year=2000, end_year=2025):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    return start + datetime.timedelta(days=random.randint(0, (end - start).days))

def generate_value(dtype):
    if dtype == 'int':
        return random.randint(0, 100)
    elif dtype == 'float':
        return round(random.uniform(0, 100), 2)
    elif dtype == 'str':
        return random_string()
    elif dtype == 'bool':
        return random.choice([True, False])
    elif dtype == 'date':
        return random_date()
    elif isinstance(dtype, dict):
        return generate_structure(dtype)
    elif isinstance(dtype, list):
        return [generate_value(t) for t in dtype]
    elif isinstance(dtype, tuple):
        return tuple(generate_value(t) for t in dtype)
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

def generate_structure(struct_def):
    if isinstance(struct_def, dict):
        return {k: generate_value(v) for k, v in struct_def.items()}
    elif isinstance(struct_def, list):
        return [generate_value(t) for t in struct_def]
    elif isinstance(struct_def, tuple):
        return tuple(generate_value(t) for t in struct_def)
    else:
        return generate_value(struct_def)

def generate_samples(num=1, **kwargs):
    structure = kwargs.get("structure", {})
    return [generate_structure(structure) for _ in range(num)]

# ---------- 作业三：修饰器添加统计功能 ----------
def stats_decorator(*metrics):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            flat_values = extract_numeric_values(data)
            results = {}
            n = len(flat_values)
            if 'SUM' in metrics:
                results['SUM'] = sum(flat_values)
            if 'AVG' in metrics:
                results['AVG'] = sum(flat_values) / n if n else 0
            if 'VAR' in metrics:
                mean = sum(flat_values) / n if n else 0
                results['VAR'] = sum((x - mean) ** 2 for x in flat_values) / n if n else 0
            if 'RMSE' in metrics:
                results['RMSE'] = math.sqrt(sum(x ** 2 for x in flat_values) / n) if n else 0
            return results
        return wrapper
    return decorator

# ---------- 工具：递归提取数值型叶节点 ----------
def extract_numeric_values(data):
    values = []
    if isinstance(data, dict):
        for v in data.values():
            values.extend(extract_numeric_values(v))
    elif isinstance(data, (list, tuple)):
        for item in data:
            values.extend(extract_numeric_values(item))
    elif isinstance(data, (int, float)):
        values.append(data)
    return values

@stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def test_sample():
    return generate_samples(
        num=100,
        structure={
            'id': 'int',
            'profile': {
                'score1': 'float',
                'score2': 'float',
                'info': {
                    'age': 'int',
                    'height': 'float'
                }
            },
            'passed': 'bool'
        }
    )

result = test_sample()
print(result)








