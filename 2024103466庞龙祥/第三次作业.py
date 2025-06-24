import random
import string
import math
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from functools import wraps


# ==================== 统计装饰器 (作业三) ====================
def stats_decorator(stats_list: List[str] = ['SUM', 'AVG', 'VAR', 'RMSE']):
    """
    为数据生成器添加统计功能的装饰器

    参数:
    stats_list -- 包含所需统计项的列表，可选值为 ['SUM', 'AVG', 'VAR', 'RMSE']
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 原始生成样本
            samples = func(*args, **kwargs)

            # 分析数据并计算统计值
            all_numbers = []
            for sample in samples:
                numbers = analyze_sample(sample)
                all_numbers.extend(numbers)

            # 计算统计值
            stats = calculate_stats(all_numbers, stats_list)

            return samples, stats

        return wrapper

    return decorator


def analyze_sample(data: Any) -> List[float]:
    """
    递归分析样本，提取所有数值型叶节点

    参数:
    data -- 要分析的数据

    返回:
    数值列表
    """
    numbers = []

    if isinstance(data, dict):
        for key, value in data.items():
            numbers.extend(analyze_sample(value))
    elif isinstance(data, list) or isinstance(data, tuple):
        for item in data:
            numbers.extend(analyze_sample(item))
    elif isinstance(data, int) or isinstance(data, float):
        numbers.append(float(data))

    return numbers


def calculate_stats(numbers: List[float], stats_list: List[str]) -> Dict[str, float]:
    """
    计算统计数据

    参数:
    numbers -- 数值列表
    stats_list -- 需要计算的统计项

    返回:
    统计结果字典
    """
    if not numbers:
        return {stat: 0.0 for stat in stats_list}

    n = len(numbers)
    total = sum(numbers)

    stats = {}

    # 计算求和
    if 'SUM' in stats_list:
        stats['SUM'] = total

    # 计算平均值
    if 'AVG' in stats_list:
        stats['AVG'] = total / n

    # 计算方差
    if 'VAR' in stats_list or 'RMSE' in stats_list:
        mean = total / n
        variance = sum((x - mean) ** 2 for x in numbers) / n

        if 'VAR' in stats_list:
            stats['VAR'] = variance

        # 计算均方根误差
        if 'RMSE' in stats_list:
            stats['RMSE'] = math.sqrt(variance)

    return stats


# ==================== 带统计功能的数据生成器 ====================
@stats_decorator()
def generate_samples(n: int = 1, **schemas) -> Tuple[List[dict], Dict[str, float]]:
    """
    生成多个样本并计算统计值

    参数:
    n -- 要生成的样本数量
    schemas -- 字段模式定义

    返回:
    包含样本列表和统计结果的元组
    """
    # 这里只是模拟数据生成，实际应用中会替换为作业二的数据生成逻辑
    samples = []
    for _ in range(n):
        sample = {
            'id': random.randint(1000, 9999),
            'name': ''.join(random.choices(string.ascii_letters, k=8)),
            'age': random.randint(18, 65),
            'salary': random.uniform(30000, 100000),
            'is_active': random.choice([True, False]),
            'scores': [random.randint(60, 100) for _ in range(5)],
            'performance': {
                'last_year': random.uniform(3.0, 5.0),
                'current_year': random.uniform(3.0, 5.0)
            },
            'location': {
                'latitude': random.uniform(-90.0, 90.0),
                'longitude': random.uniform(-180.0, 180.0)
            },
            'ratings': (random.uniform(1.0, 5.0), random.uniform(1.0, 5.0)),
            'created_at': date(2020, 1, 1) + timedelta(days=random.randint(0, 1000)),
            'projects': [
                {
                    'id': random.randint(1, 100),
                    'budget': random.uniform(1000, 10000),
                    'completed': random.choice([True, False])
                } for _ in range(random.randint(1, 3))
            ]
        }
        samples.append(sample)

    # 统计部分将由装饰器自动处理
    return samples


# ==================== 示例使用 ====================
if __name__ == "__main__":
    # 测试1: 生成用户数据并进行统计分析
    print("=== 用户数据生成与统计分析 ===")
    users, user_stats = generate_samples(
        n=100,  # 100个用户
        stats_list=['SUM', 'AVG', 'VAR', 'RMSE']  # 指定需要计算的统计项
    )

    # 打印前3个用户
    print("\n用户样本示例：")
    for i, user in enumerate(users[:3], 1):
        print(f"用户 {i}:")
        print(f"  ID: {user['id']}, 姓名: {user['name']}")
        print(f"  年龄: {user['age']}, 薪资: ${user['salary']:.2f}")
        print(f"  分数: {user['scores']}")

    # 打印统计结果
    print("\n用户数据统计分析结果：")
    for stat, value in user_stats.items():
        print(f"{stat}: {value:.4f}")

    # 测试2: 只计算求和和平均值
    print("\n=== 自定义统计项 ===")
    _, custom_stats = generate_samples(
        n=50,  # 50个用户
        stats_list=['SUM', 'AVG']  # 只计算求和和平均值
    )

    # 打印统计结果
    print("\n自定义统计分析结果：")
    for stat, value in custom_stats.items():
        print(f"{stat}: {value:.4f}")

    # 测试3: 不同类型数据的统计
    print("\n=== 不同类型数据的统计 ===")
    _, type_stats = generate_samples(
        n=200,  # 200个用户
        stats_list=['SUM', 'AVG', 'VAR']  # 指定需要计算的统计项
    )

    # 打印统计结果
    print("\n数据类型统计分析结果：")
    print(f"所有数值总和 (SUM): {type_stats['SUM']:,.2f}")
    print(f"所有数值平均值 (AVG): {type_stats['AVG']:,.4f}")
    print(f"所有数值方差 (VAR): {type_stats['VAR']:,.4f}")

    # 测试4: 复杂数据的统计
    print("\n=== 复杂数据结构统计 ===")
    _, complex_stats = generate_samples(
        n=10,  # 10个用户
        stats_list=['SUM', 'RMSE']  # 指定需要计算的统计项
    )

    # 打印统计结果
    print("\n复杂数据结构统计分析结果：")
    print(f"所有数值总和 (SUM): {complex_stats['SUM']:,.2f}")
    print(f"所有数值均方根误差 (RMSE): {complex_stats['RMSE']:,.4f}")