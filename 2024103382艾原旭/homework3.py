import random
import string
import math
from datetime import datetime, timedelta
from typing import Any, Union, List, Dict, Tuple, Callable
from functools import wraps


# 保留原有的DataSampler类
class Decorator:
    """结构化随机数据生成器（原有功能）"""

    @staticmethod
    def _random_int(**kwargs) -> int:
        min_val = kwargs.get('min', 0)
        max_val = kwargs.get('max', 100)
        return random.randint(min_val, max_val)

    @staticmethod
    def _random_float(**kwargs) -> float:
        min_val = kwargs.get('min', 0)
        max_val = kwargs.get('max', 100)
        precision = kwargs.get('precision', 2)
        return round(random.uniform(min_val, max_val), precision)

    @staticmethod
    def _random_str(**kwargs) -> str:
        length = kwargs.get('length', 8)
        chars = kwargs.get('chars', string.ascii_letters + string.digits)
        return ''.join(random.choice(chars) for _ in range(length))

    @staticmethod
    def _random_bool(**kwargs) -> bool:
        return random.choice([True, False])

    @staticmethod
    def _random_date(**kwargs) -> str:
        start = kwargs.get('start', datetime(2000, 1, 1))
        end = kwargs.get('end', datetime(2023, 12, 31))
        delta = end - start
        random_days = random.randint(0, delta.days)
        return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')

    @staticmethod
    def _random_any(**kwargs) -> Any:
        types = [int, float, str, bool, datetime]
        return Decorator._generate_value(random.choice(types), **kwargs)

    @staticmethod
    def _random_list(spec: Any, size: int, **kwargs) -> List[Any]:
        return [Decorator._generate_value(spec, **kwargs) for _ in range(size)]

    @staticmethod
    def _random_tuple(spec: Any, size: int, **kwargs) -> Tuple[Any]:
        return tuple(Decorator._random_list(spec, size, **kwargs))

    @staticmethod
    def _random_dict(spec: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return {key: Decorator._generate_value(value, **kwargs) for key, value in spec.items()}

    @staticmethod
    def _generate_value(spec: Any, **kwargs) -> Any:
        if spec is Any:
            return Decorator._random_any(**kwargs)
        elif isinstance(spec, type):
            if spec == int:
                return Decorator._random_int(**kwargs)
            elif spec == float:
                return Decorator._random_float(**kwargs)
            elif spec == str:
                return Decorator._random_str(**kwargs)
            elif spec == bool:
                return Decorator._random_bool(**kwargs)
            elif spec == datetime:
                return Decorator._random_date(**kwargs)
            else:
                raise ValueError(f"不支持的类型: {spec}")
        elif isinstance(spec, dict):
            return Decorator._random_dict(spec, **kwargs)
        elif isinstance(spec, (list, tuple)):
            container_type = list if isinstance(spec, list) else tuple
            if len(spec) != 1:
                raise ValueError("列表/元组规范应只包含一个元素，表示元素类型")
            size = kwargs.get('size', 3)
            return Decorator._random_list(spec[0], size, **kwargs) if container_type is list \
                else Decorator._random_tuple(spec[0], size, **kwargs)
        else:
            raise ValueError(f"无效的类型规范: {spec}")

    @staticmethod
    def generate_samples(spec: Any, num_samples: int = 1, **kwargs) -> Union[List[Any], Any]:
        if num_samples < 1:
            raise ValueError("样本数量必须大于0")
        samples = [Decorator._generate_value(spec, **kwargs) for _ in range(num_samples)]
        return samples[0] if num_samples == 1 else samples


# 新增的统计分析装饰器和功能
def stats_decorator(*stats_args):
    """
    带参数的装饰器，用于添加统计分析功能
    支持的统计项: 'SUM', 'AVG', 'VAR', 'RMSE'
    """
    supported_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}

    # 验证传入的统计项是否有效
    for stat in stats_args:
        if stat not in supported_stats:
            raise ValueError(f"不支持的统计项: {stat}. 支持的统计项: {supported_stats}")

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原始函数获取数据
            samples = func(*args, **kwargs)

            # 确保结果总是列表形式，方便统一处理
            samples_list = samples if isinstance(samples, list) else [samples]

            # 分析数据并计算统计量
            stats_result = {}
            numeric_values = []

            # 递归收集所有数值型叶节点
            def collect_numbers(data):
                if isinstance(data, (int, float)):
                    numeric_values.append(data)
                elif isinstance(data, dict):
                    for v in data.values():
                        collect_numbers(v)
                elif isinstance(data, (list, tuple)):
                    for item in data:
                        collect_numbers(item)

            for sample in samples_list:
                collect_numbers(sample)

            # 计算各项统计量
            if numeric_values:
                n = len(numeric_values)
                total = sum(numeric_values)
                mean = total / n

                if 'SUM' in stats_args:
                    stats_result['SUM'] = total
                if 'AVG' in stats_args:
                    stats_result['AVG'] = mean
                if 'VAR' in stats_args or 'RMSE' in stats_args:
                    squared_diffs = [(x - mean) ** 2 for x in numeric_values]
                    variance = sum(squared_diffs) / n
                    if 'VAR' in stats_args:
                        stats_result['VAR'] = variance
                    if 'RMSE' in stats_args:
                        stats_result['RMSE'] = math.sqrt(variance)

            # 将统计结果附加到原始数据上
            if isinstance(samples, list):
                return {'samples': samples, 'stats': stats_result}
            else:
                return {'sample': samples, 'stats': stats_result}

        return wrapper

    return decorator


# 使用示例
if __name__ == "__main__":
    # 1. 定义数据结构规范
    user_spec = {
        "id": int,
        "name": str,
        "age": int,
        "is_active": bool,
        "balance": float,
        "transactions": [float],  # 交易金额列表
        "preferences": {
            "min_amount": float,
            "max_amount": float
        }
    }


    # 2. 应用装饰器（统计SUM, AVG, VAR, RMSE）
    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_user_data(spec, num_samples=1, **kwargs):
        return Decorator.generate_samples(spec, num_samples, **kwargs)


    # 3. 生成带统计结果的数据
    result = generate_user_data(user_spec, num_samples=5,
                                int__min=18, int__max=80,
                                float__min=0, float__max=1000,
                                transactions__size=3)

    print("生成的样本及统计结果:")
    import pprint

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(result)


    # 4. 测试不同统计项组合
    @stats_decorator('SUM', 'AVG')
    def generate_simple_stats(spec, num_samples=1, **kwargs):
        return Decorator.generate_samples(spec, num_samples, **kwargs)


    simple_result = generate_simple_stats(user_spec, num_samples=3)
    print("\n仅含SUM和AVG的统计结果:")
    pp.pprint(simple_result)