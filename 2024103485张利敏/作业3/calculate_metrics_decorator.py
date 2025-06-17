import random
import string
import math
from functools import wraps

def calculate_metrics_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        metrics = kwargs.pop('metrics', None)
        data = func(*args, **kwargs)
        if metrics is not None:
            numbers = collect_numbers(data)
            stats = compute_stats(numbers, metrics)
            return (data, stats)
        return data
    return wrapper

def collect_numbers(data):
    numbers = []
    def _helper(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _helper(v)
        elif isinstance(obj, list):
            for item in obj:
                _helper(item)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            numbers.append(obj)
    _helper(data)
    return numbers

def compute_stats(numbers, metrics):
    stats = {}
    n = len(numbers)
    if n == 0:
        for m in metrics:
            stats[m] = None
        return stats
    total = sum(numbers)
    mean = total / n
    var = None
    if 'VAR' in metrics:
        if n >= 2:
            var = sum((x - mean)**2 for x in numbers) / (n - 1)
        else:
            var = None
    rmse = None
    if 'RMSE' in metrics:
        sum_squares = sum(x**2 for x in numbers)
        rmse = math.sqrt(sum_squares / n)
    for m in metrics:
        if m == 'SUM':
            stats[m] = total
        elif m == 'AVG':
            stats[m] = mean
        elif m == 'VAR':
            stats[m] = var
        elif m == 'RMSE':
            stats[m] = rmse
        else:
            raise ValueError(f"Unsupported metric: {m}")
    return stats

@calculate_metrics_decorator
def generate_random_data(num, **kwargs):
    result = []
    for _ in range(num):
        new_dict = {}
        for key, value in kwargs.items():
            new_dict[key] = generate_random_value(value)
        result.append(new_dict)
    return result

def generate_random_value(value):
    if isinstance(value, dict):
        return {k: generate_random_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [generate_random_value(item) for item in value]
    elif isinstance(value, str):
        length = random.randint(1, 10)
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    elif isinstance(value, (int, float)):
        return random.uniform(value*0.5, value*1.5) if isinstance(value, float) else random.randint(int(value*0.5), int(value*1.5))
    else:
        return value

# 示例使用
input_data_structure = {
    "user_info": {
        "basic": {
            "name": "张三",
            "age": 30,
            "is_vip": True
        },
        "contact": {
            "phone": {
                "mobile": "138-1234-5678",
                "home": "010-6789-1234"
            },
            "email": ["zhangsan@example.com", "backup@example.com"]
        },
        "address": {
            "country": "中国",
            "province": {
                "name": "广东省",
                "city": {
                    "name": "深圳市",
                    "district": "南山区",
                    "postcode": 518000
                }
            }
        }
    },
    "orders": [
        {
            "order_id": "O20230928001",
            "items": [
                {"product": "手机", "price": 5999.0, "spec": {"color": "黑色", "memory": "256GB"}},
                {"product": "耳机", "price": 899.0}
            ],
            "payment": {"method": "信用卡", "status": "已支付"}
        },
        {
            "order_id": "O20230928002",
            "items": [
                {"product": "智能手表", "price": 1999.0}
            ],
            "payment": {"method": "余额", "status": "待支付"}
        }
    ],
    "system_metadata": {
        "created_at": "2023-09-28T14:30:00Z",
        "updated_by": "admin01",
        "tags": ["高频客户", "电子产品"]
    }
}

# 生成数据并计算统计指标，通过修改metrics可以自定义输出统计指标
data, stats = generate_random_data(num=10, metrics=['AVG', 'SUM', 'VAR', 'RMSE'], **input_data_structure)

# 打印前两个生成的数据样本
print("Generated Data Samples:")
for item in data[:2]:
    print(item)

# 打印统计结果
print("\nStatistics:")
print(stats)
