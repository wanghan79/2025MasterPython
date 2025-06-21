import random
import string
import datetime
import math
import functools

# ----------------- 作业二：数据生成器 ----------------- #

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_date(start_year=2000, end_year=2025):
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    delta_days = (end_date - start_date).days
    return start_date + datetime.timedelta(days=random.randint(0, delta_days))

def generate_value(value_spec):
    if isinstance(value_spec, str):
        if value_spec == 'int':
            return random.randint(0, 100)
        elif value_spec == 'float':
            return round(random.uniform(0, 100), 2)
        elif value_spec == 'str':
            return random_string()
        elif value_spec == 'bool':
            return random.choice([True, False])
        elif value_spec == 'date':
            return random_date()
        else:
            raise ValueError(f"Unsupported basic type: {value_spec}")
    
    elif isinstance(value_spec, list):
        if value_spec[0] == 'list':
            return [generate_value(value_spec[1]) for _ in range(value_spec[2])]
        elif value_spec[0] == 'tuple':
            return tuple(generate_value(sub) for sub in value_spec[1:])
        else:
            raise ValueError(f"Unsupported list structure: {value_spec}")
    
    elif isinstance(value_spec, dict):
        return {k: generate_value(v) for k, v in value_spec.items()}
    
    else:
        raise TypeError(f"Unsupported spec type: {type(value_spec)}")

def data_sampler(sample_count=1, **structure_spec):
    samples = []
    for _ in range(sample_count):
        sample = {key: generate_value(spec) for key, spec in structure_spec.items()}
        samples.append(sample)
    return samples

# ----------------- 作业三：统计修饰器 ----------------- #

def stats_decorator(*stat_keys):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            flat_numbers = extract_numeric_values(data)

            result = {}
            if "SUM" in stat_keys:
                result["SUM"] = sum(flat_numbers)
            if "AVG" in stat_keys:
                result["AVG"] = sum(flat_numbers) / len(flat_numbers) if flat_numbers else 0
            if "VAR" in stat_keys:
                mean = sum(flat_numbers) / len(flat_numbers) if flat_numbers else 0
                result["VAR"] = sum((x - mean) ** 2 for x in flat_numbers) / len(flat_numbers) if flat_numbers else 0
            if "RMSE" in stat_keys:
                mean = sum(flat_numbers) / len(flat_numbers) if flat_numbers else 0
                result["RMSE"] = math.sqrt(sum((x - mean) ** 2 for x in flat_numbers) / len(flat_numbers)) if flat_numbers else 0

            return {
                "data": data,
                "stats": result
            }
        return wrapper
    return decorator

def extract_numeric_values(obj):
    nums = []

    if isinstance(obj, dict):
        for v in obj.values():
            nums.extend(extract_numeric_values(v))
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for item in obj:
            nums.extend(extract_numeric_values(item))
    elif isinstance(obj, (int, float)):
        nums.append(obj)
    
    return nums

# ----------------- 示例运行入口 main() ----------------- #

@stats_decorator("SUM", "AVG", "VAR", "RMSE")
def generate_and_analyze():
    schema = {
        "id": "int",
        "score": "float",
        "active": "bool",
        "profile": {
            "age": "int",
            "balance": "float",
            "tags": ["list", "str", 2]
        },
        "logins": ["list", {"ts": "date", "duration": "float"}, 3],
        "device": ["tuple", "str", "bool", "int"]
    }
    return data_sampler(5, **schema)

def main():
    result = generate_and_analyze()

    print("生成的数据样本：")
    for i, sample in enumerate(result["data"]):
        print(f"Sample {i+1}: {sample}")
    
    print("\n数值统计分析：")
    for k, v in result["stats"].items():
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    main()
