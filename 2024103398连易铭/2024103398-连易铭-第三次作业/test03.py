import random
import string
import numpy as np
from functools import wraps
from typing import Callable, Any, Dict, List, Literal
from abc import ABC, abstractmethod
from data_sampling import create_random_object  # 假设这是你自己写的随机对象生成函数


# ====================== 第一步：定义统计方法的接口 ======================

class BaseStatMethod(ABC):
    """
    所有统计方法都要继承这个类，实现 calculate 方法
    """
    @abstractmethod
    def calculate(self, values: List[float]) -> float:
        pass


# ====================== 第二步：具体的统计方法实现 ======================

class MeanStat(BaseStatMethod):
    def calculate(self, values: List[float]) -> float:
        return float(np.mean(values))


class VarianceStat(BaseStatMethod):
    def calculate(self, values: List[float]) -> float:
        return float(np.var(values))


class RMSEStat(BaseStatMethod):
    def calculate(self, values: List[float]) -> float:
        return float(np.sqrt(np.mean(np.square(values))))


class SumStat(BaseStatMethod):
    def calculate(self, values: List[float]) -> float:
        return float(np.sum(values))


# ====================== 第三步：工厂类用来选择具体方法 ======================

class StatMethodFactory:
    """
    根据传入的字符串类型，返回对应的统计类实例
    """
    _method_map = {
        'mean': MeanStat(),
        'variance': VarianceStat(),
        'rmse': RMSEStat(),
        'sum': SumStat()
    }

    @classmethod
    def get_method(cls, name: str) -> BaseStatMethod:
        if name not in cls._method_map:
            raise ValueError(f"不支持的统计类型：{name}")
        return cls._method_map[name]


# ====================== 第四步：生成数据样本 ======================

def generate_samples(n: int = 1, **rules):
    """
    调用 create_random_object 来生成多个样本
    """
    return [create_random_object(**rules) for _ in range(n)]


# ====================== 第五步：递归提取数字字段 ======================

def extract_numeric_fields(obj: Any, path: str = '') -> Dict[str, List[float]]:
    """
    把对象里所有数值字段都提取出来，返回一个字典 {字段路径: [值]}
    """
    result = {}

    if isinstance(obj, (int, float)):
        result[path or 'value'] = [float(obj)]

    elif isinstance(obj, dict):
        for key, val in obj.items():
            new_path = f"{path}.{key}" if path else key
            sub_result = extract_numeric_fields(val, new_path)
            result.update(sub_result)

    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            sub_result = extract_numeric_fields(item, new_path)
            result.update(sub_result)

    return result


# ====================== 第六步：装饰器，自动统计函数输出 ======================

def stat_decorator(stat_type: Literal['mean', 'variance', 'rmse', 'sum']):
    def real_decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            if not isinstance(results, (list, tuple)):
                results = [results]

            # 提取所有数值字段
            all_values: Dict[str, List[float]] = {}
            for item in results:
                extracted = extract_numeric_fields(item)
                for field, values in extracted.items():
                    all_values.setdefault(field, []).extend(values)

            # 获取方法并计算
            method = StatMethodFactory.get_method(stat_type)
            final_stats = {}
            for field, nums in all_values.items():
                final_stats[field] = method.calculate(nums) if nums else None

            return final_stats

        return wrapper
    return real_decorator


# ====================== 第七步：测试函数 ======================

@stat_decorator('mean')
def get_integers_mean(n=5):
    return generate_samples(n, value=int)


@stat_decorator('variance')
def get_integers_variance(n=5):
    return generate_samples(n, value=int)


@stat_decorator('rmse')
def get_integers_rmse(n=5):
    return generate_samples(n, value=int)


@stat_decorator('sum')
def get_integers_sum(n=5):
    return generate_samples(n, value=int)


@stat_decorator('mean')
def get_complex_objects(n=3):
    return generate_samples(
        n,
        age=int,
        score=float,
        name=str,
        nested={
            "nested_age": int,
            "nested_name": str
        }
    )


# ====================== 第八步：运行测试 ======================

if __name__ == "__main__":
    print("整数均值:", get_integers_mean(5))
    print("整数方差:", get_integers_variance(5))
    print("整数RMSE:", get_integers_rmse(5))
    print("整数求和:", get_integers_sum(5))
    print("复杂对象均值:", get_complex_objects(3))
