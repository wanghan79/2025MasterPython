import random
import string
from math import sqrt
from functools import wraps
from pprint import pprint

# ----------- 修饰器定义 -----------
def stat_decorator(*stats_to_compute):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            numeric_values = []

            # 递归提取所有数值型（int, float）数据
            def extract_numbers(obj):
                if isinstance(obj, (int, float)):
                    numeric_values.append(obj)
                elif isinstance(obj, (list, tuple, set)):
                    for item in obj:
                        extract_numbers(item)
                elif isinstance(obj, dict):
                    for value in obj.values():
                        extract_numbers(value)

            for sample in samples:
                extract_numbers(sample)

            result = {}
            if not numeric_values:
                return {stat: None for stat in stats_to_compute}

            n = len(numeric_values)
            total = sum(numeric_values)
            mean = total / n
            var = sum((x - mean) ** 2 for x in numeric_values) / n
            rmse = sqrt(sum(x ** 2 for x in numeric_values) / n)

            if 'SUM' in stats_to_compute:
                result['SUM'] = round(total, 4)
            if 'AVG' in stats_to_compute:
                result['AVG'] = round(mean, 4)
            if 'VAR' in stats_to_compute:
                result['VAR'] = round(var, 4)
            if 'RMSE' in stats_to_compute:
                result['RMSE'] = round(rmse, 4)

            return result
        return wrapper
    return decorator

# ----------- 原始样本生成函数 -----------
def random_str(length=5):
    return ''.join(random.choices(string.ascii_letters, k=length))

def random_int():
    return random.randint(0, 100)

def random_float():
    return round(random.uniform(0, 100), 2)

def random_bool():
    return random.choice([True, False])

@stat_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def generate_samples(**kwargs):
    structure = kwargs.get('structure')
    num = kwargs.get('num', 1)

    def generate_value(schema):
        if isinstance(schema, dict):
            return {key: generate_value(value) for key, value in schema.items()}
        elif isinstance(schema, (list, tuple, set)) and len(schema) == 1:
            container_type = type(schema)
            generated = [generate_value(schema[0]) for _ in range(random.randint(1, 3))]
            return container_type(generated)
        elif schema == int:
            return random_int()
        elif schema == float:
            return random_float()
        elif schema == str:
            return random_str()
        elif schema == bool:
            return random_bool()
        else:
            return None

    return [generate_value(structure) for _ in range(num)]

# ----------- 示例调用 -----------
if __name__ == "__main__":
    sample_structure = {
        "user": {
            "name": str,
            "age": int,
            "is_active": bool,
            "tags": [str],
        },
        "location": (float, float),
        "scores": [float]
    }

    result = generate_samples(structure=sample_structure, num=5)
    pprint(result)
