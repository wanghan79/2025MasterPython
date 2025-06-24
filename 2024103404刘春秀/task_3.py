import random
import string
import math
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable


# 随机数据生成函数
def random_int(): return random.randint(0, 1000)
def random_float(): return round(random.uniform(0, 1000), 2)
def random_str(length=6): return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
def random_bool(): return random.choice([True, False])
def random_date(start_year=2000, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return (start + timedelta(days=random.randint(0, (end - start).days))).strftime("%Y-%m-%d")

def generate_sample(template: Any) -> Any:
    if isinstance(template, list):
        return [generate_sample(template[0]) for _ in range(random.randint(1, 3))]
    elif isinstance(template, tuple):
        return tuple(generate_sample(item) for item in template)
    elif isinstance(template, dict):
        return {k: generate_sample(v) for k, v in template.items()}
    elif template == "int":
        return random_int()
    elif template == "float":
        return random_float()
    elif template == "str":
        return random_str()
    elif template == "bool":
        return random_bool()
    elif template == "date":
        return random_date()
    else:
        return None

def DataSampler(num: int = 5, **kwargs) -> list:
    if "data" not in kwargs:
        raise ValueError("必须通过 data=... 提供结构模板")
    template = kwargs["data"]
    return [generate_sample(template) for _ in range(num)]

# 装饰器定义
def stats_decorator(*metrics: str):
    """
    可组合的带参装饰器，支持 SUM、AVG、VAR、RMSE
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            nums = []

            def extract_numbers(obj):
                if isinstance(obj, dict):
                    for v in obj.values():
                        extract_numbers(v)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        extract_numbers(item)
                elif isinstance(obj, (int, float)):
                    nums.append(obj)

            for sample in data:
                extract_numbers(sample)

            results = {}
            if "SUM" in metrics:
                results["SUM"] = sum(nums)
            if "AVG" in metrics:
                results["AVG"] = sum(nums) / len(nums) if nums else 0
            if "VAR" in metrics:
                mean = sum(nums) / len(nums) if nums else 0
                results["VAR"] = sum((x - mean) ** 2 for x in nums) / len(nums) if nums else 0
            if "RMSE" in metrics:
                results["RMSE"] = math.sqrt(sum(x ** 2 for x in nums) / len(nums)) if nums else 0

            print(f"\n📊 统计结果（指标: {', '.join(metrics)}）:")
            for k, v in results.items():
                print(f"{k}: {v:.4f}")
            return data
        return wrapper
    return decorator


# 使用装饰器的示例
@stats_decorator("SUM", "AVG", "VAR", "RMSE")
def generate_user_data():
    template = {
        "id": "int",
        "score": "float",
        "profile": {
            "level": "int",
            "active": "bool",
            "joined": "date"
        },
        "history": ["float"]
    }
    return DataSampler(5, data=template)

if __name__ == "__main__":
    samples = generate_user_data()
    for i, item in enumerate(samples):
        print(f"\n样本 {i+1}: {item}")
