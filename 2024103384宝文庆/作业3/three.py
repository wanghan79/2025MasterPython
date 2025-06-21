import math
import random
import string
from datetime import date, timedelta
from pprint import pprint
import datetime

def analyze(data, stats=('SUM', 'AVG', 'VAR', 'RMSE')):
    numbers = []

    def extract_numbers(d):
        if isinstance(d, dict):
            for v in d.values():
                extract_numbers(v)
        elif isinstance(d, (list, tuple)):
            for v in d:
                extract_numbers(v)
        elif isinstance(d, (int, float)):
            numbers.append(d)

    extract_numbers(data)

    result = {}
    if not numbers:
        return {stat: 0 for stat in stats}

    if 'SUM' in stats:
        result['SUM'] = sum(numbers)
    if 'AVG' in stats:
        result['AVG'] = sum(numbers) / len(numbers)
    if 'VAR' in stats:
        avg = result.get('AVG', sum(numbers) / len(numbers))
        result['VAR'] = sum((x - avg) ** 2 for x in numbers) / len(numbers)
    if 'RMSE' in stats:
        result['RMSE'] = math.sqrt(sum(x ** 2 for x in numbers) / len(numbers))

    return result


def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_date():
    base = date(2000, 1, 1)
    return base + timedelta(days=random.randint(0, 10000))

def generate_sample(structure):
    if isinstance(structure, dict):
        return {k: generate_sample(v) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [generate_sample(structure[0]) for _ in range(random.randint(1, 3))]
    elif isinstance(structure, tuple):
        return tuple(generate_sample(v) for v in structure)
    elif structure == int:
        return random.randint(0, 100)
    elif structure == float:
        return round(random.uniform(0, 100), 2)
    elif structure == str:
        return random_string()
    elif structure == bool:
        return random.choice([True, False])
    elif structure == date:
        return random_date()
    else:
        return None

def DataSampler(structure, count=5):
    return [generate_sample(structure) for _ in range(count)]


if __name__ == '__main__':

    structure = {
        "id": int,
    "name": str,
    "scores": [float],
    "is_active": bool,
    "birth": datetime.date,
    "profile": {
        "height": float,
        "weight": float,
        "tags": (str, str)
    }
    }

    # 生成数据
    data = DataSampler(structure, count=3)
    print("生成的数据：")
    pprint(data)

    # 统计分析
    result = analyze(data, stats=('SUM', 'AVG', 'VAR', 'RMSE'))
    print("\n统计结果：")
    pprint(result)

