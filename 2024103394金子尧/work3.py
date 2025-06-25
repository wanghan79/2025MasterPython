
import random
import string
import datetime
import math
from typing import Any, List, Union


# -----------------------
# 工具函数：数据生成器（作业二）
# -----------------------

def random_str(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_date(start_year=2000, end_year=2023):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = (end - start).days
    return (start + datetime.timedelta(days=random.randint(0, delta))).isoformat()

def generate_field(data_type: Any) -> Any:
    if isinstance(data_type, dict):
        return {k: generate_field(v) for k, v in data_type.items()}
    elif isinstance(data_type, list):
        return [generate_field(data_type[0]) for _ in range(random.randint(1, 3))]
    elif isinstance(data_type, tuple):
        return tuple(generate_field(t) for t in data_type)
    elif data_type == int:
        return random.randint(0, 100)
    elif data_type == float:
        return round(random.uniform(0, 100), 2)
    elif data_type == str:
        return random_str()
    elif data_type == bool:
        return random.choice([True, False])
    elif data_type == 'date':
        return random_date()
    else:
        return None

def data_sampler(schema: dict, **kwargs):
    num = kwargs.get("num", 1)
    return [generate_field(schema) for _ in range(num)]


# -----------------------
# 修饰器定义（作业三）
# -----------------------

def stats_decorator(*metrics):
    def decorator(func):
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            all_values = []

            def extract_numbers(obj):
                if isinstance(obj, dict):
                    for v in obj.values():
                        extract_numbers(v)
                elif isinstance(obj, list) or isinstance(obj, tuple):
                    for item in obj:
                        extract_numbers(item)
                elif isinstance(obj, (int, float)):
                    all_values.append(obj)

            for item in data:
                extract_numbers(item)

            stats_result = {}
            n = len(all_values)
            if n == 0:
                return {metric: None for metric in metrics}

            total = sum(all_values)
            mean = total / n
            var = sum((x - mean) ** 2 for x in all_values) / n
            rmse = math.sqrt(sum(x ** 2 for x in all_values) / n)

            for metric in metrics:
                if metric.lower() == "sum":
                    stats_result["sum"] = total
                elif metric.lower() == "avg":
                    stats_result["avg"] = mean
                elif metric.lower() == "var":
                    stats_result["var"] = var
                elif metric.lower() == "rmse":
                    stats_result["rmse"] = rmse

            return stats_result
        return wrapper
    return decorator


# -----------------------
# 示例使用
# -----------------------

sample_schema = {
    "id": int,
    "name": str,
    "active": bool,
    "signup_date": "date",
    "profile": {
        "age": int,
        "height": float,
        "tags": [str],
        "location": (float, float)
    },
    "login_history": [
        {
            "login_time": "date",
            "ip": str,
            "success": bool
        }
    ]
}


@stats_decorator("sum", "avg", "var", "rmse")
def generate_user_data():
    return data_sampler(sample_schema, num=10)


if __name__ == "__main__":
    result = generate_user_data()
    print("统计结果:")
    for k, v in result.items():
        print(f"{k.upper()}: {v:.4f}")
