import functools
import math
from typing import Any, Callable, Dict, List, Set, Tuple, Union
from collections import defaultdict
import inspect


class StatsDecorator:
    """
    统计修饰器类

    功能:
    - 对被修饰函数生成的数据进行统计分析
    - 支持SUM(求和)、AVG(均值)、VAR(方差)、RMSE(均方根差)
    - 可以任意组合统计项
    """

    def __init__(self, stats: Union[str, List[str]] = "ALL"):
        """
        初始化修饰器

        参数:
            stats: 统计项,可以是字符串或列表,可选值:
                'SUM', 'AVG', 'VAR', 'RMSE', 'ALL'
        """
        if isinstance(stats, str):
            if stats == "ALL":
                self.stats = ["SUM", "AVG", "VAR", "RMSE"]
            else:
                self.stats = [stats]
        else:
            self.stats = stats

        # 验证统计项是否有效
        valid_stats = {"SUM", "AVG", "VAR", "RMSE"}
        for stat in self.stats:
            if stat not in valid_stats:
                raise ValueError(f"Invalid stat: {stat}. Valid stats are: {valid_stats}")

    def __call__(self, func: Callable) -> Callable:
        """
        修饰器调用方法
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Tuple[Any, Dict[str, Dict[str, float]]]:
            # 调用原始函数获取数据
            samples = func(*args, **kwargs)

            # 分析数据并计算统计量
            stats_results = self.analyze(samples)

            return samples, stats_results

        return wrapper

    def analyze(self, data: Any) -> Dict[str, Dict[str, float]]:
        """
        分析数据并计算统计量

        参数:
            data: 要分析的数据,可以是任意嵌套结构

        返回:
            统计结果字典,格式为:
            {
                "SUM": {"int": 123, "float": 45.6, ...},
                "AVG": {"int": 12.3, "float": 4.56, ...},
                ...
            }
        """
        # 收集所有数值型数据
        numeric_values = self._collect_numeric_values(data)

        # 计算统计量
        results = {}

        if "SUM" in self.stats:
            results["SUM"] = {k: sum(v) for k, v in numeric_values.items()}

        if "AVG" in self.stats:
            results["AVG"] = {k: sum(v) / len(v) if v else 0 for k, v in numeric_values.items()}

        if "VAR" in self.stats or "RMSE" in self.stats:
            # 计算方差需要先计算平均值
            avg_results = {}
            for dtype, values in numeric_values.items():
                if values:
                    avg = sum(values) / len(values)
                    avg_results[dtype] = avg
                    variance = sum((x - avg) ** 2 for x in values) / len(values)
                    results.setdefault("VAR", {})[dtype] = variance
                    results.setdefault("RMSE", {})[dtype] = math.sqrt(variance)
                else:
                    results.setdefault("VAR", {})[dtype] = 0
                    results.setdefault("RMSE", {})[dtype] = 0

        # 只保留请求的统计项
        return {k: v for k, v in results.items() if k in self.stats}

    def _collect_numeric_values(self, data: Any) -> Dict[str, List[Union[int, float]]]:
        """
        递归收集所有数值型数据

        返回:
            按类型分组的数值字典,格式为:
            {
                "int": [1, 2, 3, ...],
                "float": [1.1, 2.2, ...]
            }
        """
        numeric_values = defaultdict(list)

        if isinstance(data, (int, float)):
            dtype = "int" if isinstance(data, int) else "float"
            numeric_values[dtype].append(data)
        elif isinstance(data, (list, tuple)):
            for item in data:
                nested_values = self._collect_numeric_values(item)
                for dtype, values in nested_values.items():
                    numeric_values[dtype].extend(values)
        elif isinstance(data, dict):
            for value in data.values():
                nested_values = self._collect_numeric_values(value)
                for dtype, values in nested_values.items():
                    numeric_values[dtype].extend(values)

        return numeric_values


def demonstrate_stats_decorator():
    """
    演示StatsDecorator的功能
    """
    from data_sampler import DataSampler

    # 创建数据采样器实例
    sampler = DataSampler()

    # 使用修饰器生成简单数据并计算所有统计量
    @StatsDecorator("ALL")
    def generate_simple_data(count: int):
        return sampler.generate_samples(
            count,
            id="int",
            temperature="float",
            readings={
                "type": "list",
                "length": 5,
                "element_type": "float"
            }
        )

    print("1: 简单数据统计")
    samples, stats = generate_simple_data(10)
    print("统计结果:")
    for stat, values in stats.items():
        print(f"{stat}:")
        for dtype, value in values.items():
            print(f"  {dtype}: {value:.4f}" if isinstance(value, float) else f"  {dtype}: {value}")

    # 使用修饰器生成复杂数据并选择特定统计量
    @StatsDecorator(["SUM", "AVG"])
    def generate_complex_data(count: int):
        return sampler.generate_samples(
            count,
            user_id="int",
            transactions={
                "type": "list",
                "length": 3,
                "element_type": {
                    "type": "dict",
                    "structure": {
                        "amount": "float",
                        "items": {
                            "type": "list",
                            "length": 2,
                            "element_type": {
                                "type": "dict",
                                "structure": {
                                    "price": "float",
                                    "quantity": "int"
                                }
                            }
                        }
                    }
                }
            }
        )

    print("\n2: 复杂数据统计(SUM和AVG)")
    samples, stats = generate_complex_data(5)
    print("统计结果:")
    for stat, values in stats.items():
        print(f"{stat}:")
        for dtype, value in values.items():
            print(f"  {dtype}: {value:.4f}" if isinstance(value, float) else f"  {dtype}: {value}")

    # 直接使用analyze方法分析现有数据
    print("\n3: 直接分析现有数据")
    existing_data = [
        {"id": 1, "values": [10, 20.5, 30]},
        {"id": 2, "values": [15, 25.5, {"nested": 35.5}]},
        {"id": 3, "values": (20, 30.5, [40, 45.5])}
    ]

    decorator = StatsDecorator(["SUM", "VAR", "RMSE"])
    stats = decorator.analyze(existing_data)

    print("统计结果:")
    for stat, values in stats.items():
        print(f"{stat}:")
        for dtype, value in values.items():
            print(f"  {dtype}: {value:.4f}" if isinstance(value, float) else f"  {dtype}: {value}")


if __name__ == "__main__":
    demonstrate_stats_decorator()
