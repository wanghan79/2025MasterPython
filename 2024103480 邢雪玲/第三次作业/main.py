from functools import wraps
import math
import time
from datetime import datetime, timedelta
import random


class StatsCalculator:
    @classmethod
    def avg(cls, data: list) -> float:
        return sum(data) / len(data) if data else 0

    @classmethod
    def var(cls, data: list) -> float:
        mean = cls.avg(data)
        return sum((x - mean) ** 2 for x in data) / len(data) if data else 0

    @classmethod
    def rmse(cls, data: list) -> float:
        return math.sqrt(cls.var(data))

    @classmethod
    def sum(cls, data: list) -> float:
        return sum(data)

    @classmethod
    def get_stat_methods(cls) -> dict:
        return {
            'avg': cls.avg,
            'var': cls.var,
            'rmse': cls.rmse,
            'sum': cls.sum
        }


def statistical_metrics(*metrics):
    """
    通用统计装饰器工厂函数
    :param metrics: 可变参数，接收需要计算的统计指标名称
    :return: 装饰器函数
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            raw_data = func(*args, **kwargs)

            def extract_numbers(data):
                if isinstance(data, (int, float)):
                    return [data]
                if isinstance(data, dict):
                    return [num for v in data.values() for num in extract_numbers(v)]
                if isinstance(data, (list, tuple)):
                    return [num for item in data for num in extract_numbers(item)]
                return []

            numbers = extract_numbers(raw_data)

            calculator = StatsCalculator()
            valid_metrics = {}
            for metric in metrics:
                if hasattr(calculator, metric):
                    valid_metrics[metric] = getattr(calculator, metric)(numbers)

            return {
                'timestamp': datetime.now().isoformat(),
                'raw_data': raw_data,
                'statistics': valid_metrics
            }

        return wrapper

    return decorator


@statistical_metrics('avg', 'var', 'rmse', 'sum')
def generate_time_series_samples(days: int, points_per_day: int) -> list:
    """生成带噪声的时间序列数据"""
    base_date = datetime(2025, 5, 1)
    data = []

    for day in range(days):
        daily_data = {
            'date': (base_date + timedelta(days=day)).strftime("%Y-%m-%d"),
            'readings': [
                {
                    'time': f"{hour:02d}:{minute:02d}",
                    'value': 50 + 10 * math.sin(hour) + random.gauss(0, 2)
                }
                for hour in range(24)
                for minute in random.sample(range(0, 60), points_per_day)
            ]
        }
        data.append(daily_data)

    return data


# 测试执行
if __name__ == "__main__":
    result = generate_time_series_samples(days=3, points_per_day=3)

    print(f"Statistics at {result['timestamp']}")
    for metric, value in result['statistics'].items():
        print(f"{metric.upper():<6}: {value:.4f}")

    # 原始数据结构示例
    print("\nSample raw data structure:")
    print(f"Days generated: {len(result['raw_data'])}")
    print(f"First day readings: {len(result['raw_data'][0]['readings'])} points")
    print(f"Sample value: {result['raw_data'][0]['readings'][0]['value']:.2f}")
