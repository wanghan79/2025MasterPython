import math
from functools import wraps

# 修饰器
def stats_decorator(func):
    @wraps(func)
    def wrapper(data):
        print(f"计算 {func.__name__} 中...")
        return func(data)
    return wrapper

@stats_decorator
def SUM(data): return sum(data)

@stats_decorator
def AVG(data): return sum(data) / len(data)

@stats_decorator
def VAR(data):
    mean = AVG(data)
    return sum((x - mean) ** 2 for x in data) / len(data)

@stats_decorator
def RMSE(data):
    mean = AVG(data)
    return math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))

def analyze(data):
    return {
        "sum": SUM(data),
        "avg": AVG(data),
        "var": VAR(data),
        "rmse": RMSE(data)
    }

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    print(analyze(data))
