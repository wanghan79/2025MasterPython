import math
from functools import wraps
from hw2 import generate_samples  # 导入你自己封装的函数

# 修饰器工厂：接收若干统计项参数
def stat_decorator(*stats):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            numeric_values = extract_numbers(data)

            results = {}
            if not numeric_values:
                return {"stats": {}, "data": data}

            if "SUM" in stats:
                results["SUM"] = sum(numeric_values)
            if "AVG" in stats:
                results["AVG"] = sum(numeric_values) / len(numeric_values)
            if "VAR" in stats:
                mean = sum(numeric_values) / len(numeric_values)
                results["VAR"] = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
            if "RMSE" in stats:
                mean = sum(numeric_values) / len(numeric_values)
                mse = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
                results["RMSE"] = math.sqrt(mse)

            return {
                "stats": results,
                "data": data
            }
        return wrapper
    return decorator

def extract_numbers(data):
    values = []
    if isinstance(data, (int, float)):
        values.append(data)
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            values.extend(extract_numbers(item))
    elif isinstance(data, dict):
        for val in data.values():
            values.extend(extract_numbers(val))
    return values

# 用修饰器修饰 generate_samples
@stat_decorator("SUM", "AVG", "VAR", "RMSE")
def decorated_samples(**kwargs):
    return generate_samples(**kwargs)

structure = {
        "name": str,
        "age": int,
        "scores": [float],
        "nested": {
            "height": float,
            "married": bool
        }
    }

result = decorated_samples(count=5, structure=structure)
print("统计结果：", result["stats"])
print("样本示例：", result["data"][0])
