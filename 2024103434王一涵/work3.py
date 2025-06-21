from functools import wraps
from typing import Callable, Dict, List, Any, Union
from work2 import generate_samples

def stats_decorator(stats: List[str]) -> Callable:
    """
    统计嵌套数据中数值型数据的装饰器
    参数:
        stats: 需要统计的指标列表 (可选['sum', 'avg', 'var', 'rmse'])
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # 生成原始样本集
            samples = func(*args, **kwargs)

            # 递归提取所有数值型数据
            numbers = []

            def extract_numbers(data: Any) -> None:
                if isinstance(data, (int, float)):
                    numbers.append(data)
                elif isinstance(data, (list, tuple)):
                    for item in data:
                        extract_numbers(item)
                elif isinstance(data, dict):
                    for v in data.values():
                        extract_numbers(v)
            # 遍历所有样本提取数值
            for sample in samples:
                extract_numbers(sample)
            n = len(numbers)
            result = {}
            if n == 0:
                return {stat: None for stat in stats}
            # 计算基础统计量
            sum_val = sum(numbers)
            avg_val = sum_val / n
            # 按需计算指定统计量
            if 'sum' in stats:
                result['sum'] = round(sum_val, 4)
            if 'avg' in stats:
                result['avg'] = round(avg_val, 4)
            if 'var' in stats:
                var_val = sum((x - avg_val) ** 2 for x in numbers) / n
                result['var'] = round(var_val, 4)
            if 'rmse' in stats:
                # 样本标准差
                if n < 2:
                    rmse_val = 0.0
                else:
                    var_sample = sum((x - avg_val) ** 2 for x in numbers) / (n - 1)
                    rmse_val = round(var_sample ** 0.5, 4)
                result['rmse'] = rmse_val
            return result
        return wrapper
    return decorator

@stats_decorator(['sum', 'avg', 'var', 'rmse'])
def generate_numeric_samples(**kwargs) -> List[Any]:
    """生成包含数值的嵌套样本"""
    return generate_samples(num_samples=kwargs.get('num_samples', 10),
                            max_depth=kwargs.get('max_depth', 3),
                            types=[int, float])


# 测试运行
if __name__ == "__main__":
    samples = generate_numeric_samples(num_samples=5, max_depth=2)
    print("生成的样本:", samples)
    stats_result = generate_numeric_samples(num_samples=5, max_depth=2)
    print("统计结果:", stats_result)