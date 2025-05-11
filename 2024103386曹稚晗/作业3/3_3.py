import random
import string
import numpy as np
from functools import wraps
from typing import Callable, Any, Dict, List, Union, Literal
from data_sampling import create_random_object
from abc import ABC, abstractmethod


class StatisticalMethod(ABC):
    """统计方法的抽象基类"""

    @abstractmethod
    def calculate(self, values: List[float]) -> float:
        """计算统计值"""
        pass


class MeanMethod(StatisticalMethod):
    """均值统计方法"""

    def calculate(self, values: List[float]) -> float:
        return float(np.mean(values))


class VarianceMethod(StatisticalMethod):
    """方差统计方法"""

    def calculate(self, values: List[float]) -> float:
        return float(np.var(values))


class RMSEMethod(StatisticalMethod):
    """均方根误差统计方法"""

    def calculate(self, values: List[float]) -> float:
        return float(np.sqrt(np.mean(np.square(values))))


class SumMethod(StatisticalMethod):
    """求和统计方法"""

    def calculate(self, values: List[float]) -> float:
        return float(np.sum(values))


class StatisticalMethodFactory:
    """统计方法工厂类"""

    _methods = {
        'mean': MeanMethod(),
        'variance': VarianceMethod(),
        'rmse': RMSEMethod(),
        'sum': SumMethod()
    }

    @classmethod
    def get_method(cls, stat_type: str) -> StatisticalMethod:
        """获取统计方法实例"""
        if stat_type not in cls._methods:
            raise ValueError(f"Unsupported statistical type: {stat_type}")
        return cls._methods[stat_type]


def sample_generator(num_samples: int = 1, **kwargs):
    """
    生成指定数量的随机样本

    Args:
        num_samples: 要生成的样本数量
        **kwargs: 传递给create_random_object的参数

    Returns:
        生成的样本列表
    """
    return [create_random_object(**kwargs) for _ in range(num_samples)]


def extract_field_values(obj: Any, field_path: str = '') -> Dict[str, List[float]]:
    """
    从对象中提取字段值

    Args:
        obj: 输入对象，可以是数值、字典或列表
        field_path: 当前字段的路径

    Returns:
        字段路径到值的映射字典
    """
    result = {}

    if isinstance(obj, (int, float)):
        if field_path:
            result[field_path] = [float(obj)]
        else:
            result['value'] = [float(obj)]
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{field_path}.{key}" if field_path else key
            result.update(extract_field_values(value, new_path))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            new_path = f"{field_path}[{i}]" if field_path else f"[{i}]"
            result.update(extract_field_values(item, new_path))

    return result


def statistical_decorator(stat_type: Literal['mean', 'variance', 'rmse', 'sum'] = 'mean'):
    """
    统一的统计装饰器，通过参数控制不同的统计功能

    Args:
        stat_type: 统计类型，可选值：'mean', 'variance', 'rmse', 'sum'
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            if not isinstance(results, (list, tuple)):
                results = [results]

            # 提取所有字段的值
            field_values = {}
            for result in results:
                for field, values in extract_field_values(result).items():
                    if field not in field_values:
                        field_values[field] = []
                    field_values[field].extend(values)

            # 获取统计方法
            stat_method = StatisticalMethodFactory.get_method(stat_type)

            # 对每个字段进行统计
            stats = {}
            for field, values in field_values.items():
                if not values:
                    stats[field] = None
                    continue
                stats[field] = stat_method.calculate(values)

            return stats

        return wrapper

    return decorator


# 示例使用
if __name__ == "__main__":
    # 示例1：生成随机整数并计算不同统计量
    @statistical_decorator(stat_type='mean')
    def generate_random_numbers_mean(num_samples: int = 5):
        return sample_generator(num_samples, value=int)


    @statistical_decorator(stat_type='variance')
    def generate_random_numbers_var(num_samples: int = 5):
        return sample_generator(num_samples, value=int)


    @statistical_decorator(stat_type='rmse')
    def generate_random_numbers_rmse(num_samples: int = 5):
        return sample_generator(num_samples, value=int)


    @statistical_decorator(stat_type='sum')
    def generate_random_numbers_sum(num_samples: int = 5):
        return sample_generator(num_samples, value=int)


    # 示例2：生成复杂对象并计算统计量
    @statistical_decorator(stat_type='mean')
    def generate_complex_objects(num_samples: int = 5):
        return sample_generator(
            num_samples,
            age=int,
            score=float,
            name=str,
            nested=dict(
                nested_age=int,
                nested_name=str
            )
        )


    # 测试代码
    print("Mean of random numbers:", generate_random_numbers_mean(5))
    print("Variance of random numbers:", generate_random_numbers_var(5))
    print("RMSE of random numbers:", generate_random_numbers_rmse(5))
    print("Sum of random numbers:", generate_random_numbers_sum(5))
    print("\nComplex objects mean:", generate_complex_objects(3))