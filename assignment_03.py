# coding=utf-8
import random
import string
from datetime import datetime, timedelta
from functools import wraps
import math
import pprint

# ===================== 数据生成函数 =====================
def generate_samples(**kwargs) -> list:
    """
    生成任意嵌套结构的随机样本数据集

    参数:
        size: 生成的样本数量
        schema: 描述数据结构的嵌套定义

    返回:
        包含指定数量样本的列表，每个样本符合给定的数据结构
    """
    def generate_value(data_type: any) -> any:
        """根据类型描述生成随机值"""
        # 基本数据类型
        if data_type is int:
            return random.randint(1, 100)
        elif data_type is float:
            return round(random.uniform(0.1, 100.0), 2)
        elif data_type is str:
            length = random.randint(5, 10)
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        elif data_type is bool:
            return random.choice([True, False])
        elif data_type is datetime:
            start = datetime(2000, 1, 1)
            end = datetime(2023, 12, 31)
            return start + timedelta(days=random.randint(0, (end - start).days))

        # 容器类型
        elif isinstance(data_type, list):
            # 列表长度随机 (1-5个元素)
            return [generate_value(data_type[0]) for _ in range(random.randint(1, 5))]
        elif isinstance(data_type, tuple):
            return tuple(generate_value(item) for item in data_type)
        elif isinstance(data_type, dict):
            return {key: generate_value(value) for key, value in data_type.items()}

        # 自定义嵌套结构
        elif callable(data_type):
            return data_type()

        else:
            raise TypeError(f"不支持的数据类型: {type(data_type)}")

    # 验证必需参数
    if 'size' not in kwargs or 'schema' not in kwargs:
        raise ValueError("必须包含'size'和'schema'参数")

    size = kwargs['size']
    schema = kwargs['schema']

    # 生成样本数据集
    return [generate_value(schema) for _ in range(size)]


# ===================== 统计分析装饰器 =====================
def stats_decorator(*stats_args):
    """
    带参数的装饰器，用于统计分析数值型数据

    参数:
        stats_args: 统计项列表，可选值: 'SUM', 'AVG', 'VAR', 'RMSE'

    使用示例:
        @stats_decorator('SUM', 'AVG', 'VAR')
        def generate_samples(**kwargs):
            ...
    """
    # 验证统计项参数
    valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
    for stat in stats_args:
        if stat not in valid_stats:
            raise ValueError(f"无效统计项: {stat}. 可选值: {valid_stats}")

    def decorator(func):
        @wraps(func)
        def wrapper(**kwargs):
            # 1. 调用原始函数生成样本
            samples = func(**kwargs)

            # 2. 收集所有数值型数据（递归遍历嵌套结构）
            numbers = []
            def collect_numbers(data):
                """递归收集数值型数据"""
                if isinstance(data, (int, float)):
                    numbers.append(data)
                elif isinstance(data, (list, tuple)):
                    for item in data:
                        collect_numbers(item)
                elif isinstance(data, dict):
                    for value in data.values():
                        collect_numbers(value)
                # 其他类型（str, bool, datetime等）忽略

            for sample in samples:
                collect_numbers(sample)

            # 3. 计算统计指标
            results = {}
            n = len(numbers)

            if n > 0:
                total = sum(numbers)
                mean = total / n

                # 根据请求的统计项计算结果
                if 'SUM' in stats_args:
                    results['SUM'] = total

                if 'AVG' in stats_args:
                    results['AVG'] = mean

                if 'VAR' in stats_args or 'RMSE' in stats_args:
                    # 方差 = Σ(x_i - μ)^2 / n
                    variance = sum((x - mean) ** 2 for x in numbers) / n

                    if 'VAR' in stats_args:
                        results['VAR'] = variance

                    if 'RMSE' in stats_args:
                        # RMSE = sqrt(VAR)
                        results['RMSE'] = math.sqrt(variance)

            # 4. 返回样本集和统计结果
            return samples, results

        return wrapper
    return decorator


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 1. 定义复杂数据结构
    def nested_structure():
        """自定义嵌套结构生成器"""
        return {
            'matrix': [[random.randint(1, 100) for _ in range(3)] for _ in range(3)],
            'metadata': (
                random.uniform(1.0, 10.0),
                random.choice([True, False])
            )
        }

    # 2. 定义数据结构模式
    complex_schema = {
        'id': int,
        'name': str,
        'scores': [float],
        'details': {
            'active': bool,
            'values': (int, float, int),
            'history': [{
                'timestamp': datetime,
                'value': float
            }]
        },
        'extra': nested_structure  # 自定义嵌套结构
    }

    # 3. 应用装饰器（选择所有统计项）
    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_samples_with_stats(**kwargs):
        return generate_samples(**kwargs)

    # 4. 生成样本并分析
    samples, stats = generate_samples_with_stats(
        size=100,  # 生成100个样本
        schema=complex_schema
    )

    # 5. 打印结果
    print(f"生成样本数量: {len(samples)}")
    print(f"数值型数据点数量: {len(samples[0])}")  # 近似值

    print("\n统计结果:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.4f}" if isinstance(value, float) else f"{stat}: {value}")

    print("\n第一个样本示例:")
    pprint.pprint(samples[0], width=120, sort_dicts=False)