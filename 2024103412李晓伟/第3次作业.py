import random
import string
import copy
import math
from datetime import date
from functools import wraps  # 导入wraps

# 定义analyze方法
def analyze(samples):
    all_numbers = []
    for sample in samples:
        def extract_numbers(data):
            if isinstance(data, dict):
                for value in data.values():
                    extract_numbers(value)
            elif isinstance(data, list):
                for item in data:
                    extract_numbers(item)
            elif isinstance(data, (int, float)):
                all_numbers.append(data)

        extract_numbers(sample)

    results = {}
    if all_numbers:
        sum_val = sum(all_numbers)
        avg = sum_val / len(all_numbers)
        var = sum((x - avg) ** 2 for x in all_numbers) / len(all_numbers)
        rmse = math.sqrt(var)

        results['SUM'] = sum_val
        results['AVG'] = avg
        results['VAR'] = var
        results['RMSE'] = rmse

    return results

# 定义装饰器
def stats_decorator(*stats):
    def decorator(func):
        @wraps(func)  # 使用wraps装饰器
        def wrapper(**kwargs):
            samples = func(**kwargs)

            results = analyze(samples)

            # 过滤需要的统计项
            filtered_results = {stat: results[stat] for stat in stats if stat in results}

            # 输出生成的数据
            for i, sample in enumerate(samples, 1):
                print(f"Sample {i}: {sample}")

            # 输出统计结果
            print("\n统计结果:")
            for stat, value in filtered_results.items():
                print(f"{stat}: {value}")

            return samples
        return wrapper
    return decorator

# 应用装饰器
@stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def generate_random_sample(**kwargs):
    def generate_random_string(length=5):
        return ''.join(random.choice(string.ascii_letters) for _ in range(length))

    def generate_random_int():
        return random.randint(1, 100)

    def generate_random_float():
        return round(random.uniform(1.0, 100.0), 2)

    def generate_random_bool():
        return random.choice([True, False])

    def generate_random_date():
        return date.today()

    def generate_random_list(length=3, value_type='int'):
        return [generate_random_value(value_type) for _ in range(length)]

    def generate_random_dict(keys, values):
        return {key: generate_random_value(value) for key, value in zip(keys, values)}

    def generate_random_value(value_type):
        if value_type == 'str':
            return generate_random_string()
        elif value_type == 'int':
            return generate_random_int()
        elif value_type == 'float':
            return generate_random_float()
        elif value_type == 'bool':
            return generate_random_bool()
        elif value_type == 'date':
            return generate_random_date()
        elif value_type == 'list':
            return generate_random_list()
        elif isinstance(value_type, dict):
            return generate_random_dict(value_type.keys(), value_type.values())
        elif isinstance(value_type, list):
            return [generate_random_value(item) for item in value_type]
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

    data_structure = kwargs.get('data_structure')
    num_samples = kwargs.get('num_samples', 1)

    samples = []
    for i in range(num_samples):
        sample = generate_random_value(copy.deepcopy(data_structure))
        samples.append(sample)

    return samples

# 调用
data_structure = {
    'name': 'str',
    'age': 'int',
    'scores': ['int', 'int', 'int'],
    'is_male': 'bool',
    'created_time': 'date',
    'address': {
        'street': 'str',
        'number': 'int'
    }
}
num_samples = 6

samples = generate_random_sample(data_structure=data_structure, num_samples=num_samples)


# 输出：
# Sample 1: {'name': 'kUkPf', 'age': 40, 'scores': [56, 40, 92], 'is_male': False, 'created_time': datetime.date(2025, 6, 11), 'address': {'street': 'eNDGQ', 'number': 83}}
# Sample 2: {'name': 'VNERo', 'age': 33, 'scores': [29, 32, 12], 'is_male': True, 'created_time': datetime.date(2025, 6, 11), 'address': {'street': 'vxIUT', 'number': 15}}
# Sample 3: {'name': 'VofiH', 'age': 41, 'scores': [65, 15, 9], 'is_male': False, 'created_time': datetime.date(2025, 6, 11), 'address': {'street': 'gXHAO', 'number': 92}}
# Sample 4: {'name': 'SNlAJ', 'age': 51, 'scores': [67, 38, 19], 'is_male': False, 'created_time': datetime.date(2025, 6, 11), 'address': {'street': 'zGwxB', 'number': 23}}
# Sample 5: {'name': 'oqxXZ', 'age': 76, 'scores': [15, 42, 40], 'is_male': False, 'created_time': datetime.date(2025, 6, 11), 'address': {'street': 'uWzuV', 'number': 12}}
# Sample 6: {'name': 'lvwzH', 'age': 37, 'scores': [35, 22, 71], 'is_male': True, 'created_time': datetime.date(2025, 6, 11), 'address': {'street': 'hpFOA', 'number': 39}}
#
# 统计结果:
# SUM: 1243
# AVG: 34.52777777777778
# VAR: 704.0270061728395
# RMSE: 26.533507234680446
