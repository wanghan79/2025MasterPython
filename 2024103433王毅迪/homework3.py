import random
import string
import math
from functools import wraps

def generate_random_value(data_type):
 
    if data_type == int:
        return random.randint(0, 100)
    elif data_type == float:
        return round(random.uniform(0, 100), 2)
    elif data_type == str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    elif data_type == bool:
        return random.choice([True, False])
    else:
        return None

def generate_sample(structure):
 
    if isinstance(structure, dict):
        return {k: generate_sample(v) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [generate_sample(v) for v in structure]
    elif isinstance(structure, tuple):
        return tuple(generate_sample(v) for v in structure)
    else:
        return generate_random_value(structure)

def generate_samples(**kwargs):

    structure = kwargs.get('structure')
    num = kwargs.get('num', 1)

    if structure is None:
        raise ValueError("参数 'structure' 是必需的")

    return [generate_sample(structure) for _ in range(num)]

# ===== 作业3修饰器部分 =====

def statistics_decorator(*metrics):
 
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            numbers = []

            def extract_numbers(obj):
                """递归提取所有 int 和 float 类型的数值"""
                if isinstance(obj, (int, float)):
                    numbers.append(obj)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        extract_numbers(v)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        extract_numbers(item)

            for item in results:
                extract_numbers(item)

            stats = {}
            if not numbers:
                print("未找到任何数值型数据")
                return stats

            n = len(numbers)
            s = sum(numbers)
            mean = s / n

            if 'SUM' in metrics:
                stats['SUM'] = round(s, 2)
            if 'AVG' in metrics:
                stats['AVG'] = round(mean, 2)
            if 'VAR' in metrics:
                stats['VAR'] = round(sum((x - mean) ** 2 for x in numbers) / n, 2)
            if 'RMSE' in metrics:
                stats['RMSE'] = round(math.sqrt(sum(x ** 2 for x in numbers) / n), 2)

            print(f"\n统计结果（{', '.join(metrics)}）：")
            for k, v in stats.items():
                print(f"{k} = {v}")

            return results
        return wrapper
    return decorator

# ===== 示例使用 =====

@statistics_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def get_samples():
    structure = {
        'id': int,
        'name': str,
        'score': [float, float, float],
        'flag': bool,
        'profile': {
            'height': float,
            'weight': float,
            'scores': (int, int)
        }
    }
    return generate_samples(structure=structure, num=5)

if __name__ == "__main__":
    samples = get_samples()
    for i, sample in enumerate(samples):
        print(f"样本 {i+1}: {sample}")
