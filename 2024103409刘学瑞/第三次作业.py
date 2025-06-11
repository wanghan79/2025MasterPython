# stats_decorator_analysis.py
import random
import string
import math
from functools import wraps

# ------------------ 带参数的统计装饰器 -------------------
def stats_decorator(*stat_items):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            numbers = extract_numbers(result)
            stats = {}
            n = len(numbers)

            if "SUM" in stat_items:
                stats["SUM"] = sum(numbers)
            if "AVG" in stat_items:
                stats["AVG"] = sum(numbers) / n if n else 0.0
            if "VAR" in stat_items:
                mean = sum(numbers) / n if n else 0.0
                stats["VAR"] = sum((x - mean) ** 2 for x in numbers) / n if n else 0.0
            if "RMSE" in stat_items:
                mean = sum(numbers) / n if n else 0.0
                variance = sum((x - mean) ** 2 for x in numbers) / n if n else 0.0
                stats["RMSE"] = math.sqrt(variance)

            wrapper.stats_result = stats
            return result

        wrapper.stats_result = {}
        return wrapper
    return decorator

# ------------------ 工具函数 -------------------
def extract_numbers(data):
    if isinstance(data, dict):
        return [x for v in data.values() for x in extract_numbers(v)]
    elif isinstance(data, (list, tuple, set)):
        return [x for item in data for x in extract_numbers(item)]
    elif isinstance(data, (int, float)):
        return [data]
    else:
        return []

# ------------------ 数据采样类 -------------------
class DataSampling:
    @classmethod
    @stats_decorator("SUM", "AVG", "VAR", "RMSE")
    def sampling(cls, **kwargs):
        def check_type(key, value):
            if key == "int":
                return random.randint(value["datarange"][0], value["datarange"][1])
            elif key == "float":
                return random.uniform(value["datarange"][0], value["datarange"][1])
            elif key == "str":
                length = value.get("len", 10)
                return ''.join(random.choices(value["datarange"], k=length))
            elif key in ["list", "tuple", "dict"]:
                return []
            else:
                return None

        def recursive_sampling(node):
            sample = {}
            for key, value in node.items():
                if isinstance(value, dict) and "subNodes" in value:
                    sample[key] = recursive_sampling(value["subNodes"])
                else:
                    sample[key] = check_type(key, value)
            return sample

        result = []
        for key, value in kwargs.items():
            if isinstance(value, dict) and "subNodes" in value:
                result.append(recursive_sampling(value["subNodes"]))
            else:
                result.append(check_type(key, value))
        return result

    @staticmethod
    def analyze(data, stats=("SUM", "AVG", "VAR", "RMSE")):
        numbers = extract_numbers(data)
        n = len(numbers)
        result = {}

        if "SUM" in stats:
            result["SUM"] = sum(numbers)
        if "AVG" in stats:
            result["AVG"] = sum(numbers) / n if n else 0.0
        if "VAR" in stats:
            mean = sum(numbers) / n if n else 0.0
            result["VAR"] = sum((x - mean) ** 2 for x in numbers) / n if n else 0.0
        if "RMSE" in stats:
            mean = sum(numbers) / n if n else 0.0
            variance = sum((x - mean) ** 2 for x in numbers) / n if n else 0.0
            result["RMSE"] = math.sqrt(variance)

        return result

# ------------------ 示例 -------------------
if __name__ == "__main__":
    data_structure = {
        "int": {"datarange": (0, 100)},
        "float": {"datarange": (0, 10000)},
        "str": {"datarange": string.ascii_uppercase, "len": 5},
        "list": {"subNodes": {"int": {"datarange": (0, 10)}}},
        "tuple": {"subNodes": {"float": {"datarange": (0, 1000)}}},
        "dict": {"subNodes": {"str": {"datarange": string.ascii_uppercase, "len": 3}}}
    }

    samples = DataSampling.sampling(**data_structure)

    print("生成的样本数据:")
    for sample in samples:
        print(sample)

    print("\n装饰器统计结果:")
    for key, value in DataSampling.sampling.stats_result.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nAnalyze 分析结果:")
    analysis = DataSampling.analyze(samples, stats=("SUM", "AVG", "VAR", "RMSE"))
    for key, value in analysis.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

