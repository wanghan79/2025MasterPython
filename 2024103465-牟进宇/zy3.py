import math
from typing import List, Dict, Any, Tuple  # 添加Tuple导入

from zy2 import generate_samples

def stats_decorator(stats: List[str]):
    """
    带参数装饰器，用于统计样本集中的数值型数据

    Args:
        stats: 统计项列表（支持'SUM'/'AVG'/'VAR'/'RMSE'任意组合）
    """
    # 验证统计项有效性
    valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
    for stat in stats:
        if stat not in valid_stats:
            raise ValueError(f"无效统计项: {stat}（支持项：{valid_stats}）")

    def decorator(func):
        # 修改返回类型注解为Tuple
        def wrapper(*args, **kwargs) -> Tuple[List[Any], Dict[str, float]]:
            # 生成样本集
            samples = func(*args, **kwargs)

            # 递归提取所有数值（int/float）
            def _extract_numbers(data) -> List[float]:
                numbers = []
                if isinstance(data, (list, tuple)):  # 处理列表/元组
                    for item in data:
                        numbers.extend(_extract_numbers(item))
                elif isinstance(data, dict):  # 处理字典
                    for value in data.values():
                        numbers.extend(_extract_numbers(value))
                elif isinstance(data, (int, float)):  # 收集数值
                    numbers.append(float(data))
                return numbers

            all_numbers = _extract_numbers(samples)
            n = len(all_numbers)
            results = {}

            # 计算统计项（避免重复计算）
            if n == 0:
                return samples, results  # 无数值时返回空结果

            # 计算总和（SUM）
            if 'SUM' in stats:
                total = sum(all_numbers)
                results['SUM'] = total

            # 计算均值（AVG）
            if 'SUM' in results:
                avg = results['SUM'] / n
            else:
                avg = sum(all_numbers) / n
            if 'AVG' in stats:
                results['AVG'] = avg

            # 计算方差（VAR）
            if 'VAR' in stats:
                var = sum((x - avg)**2 for x in all_numbers) / n
                results['VAR'] = var

            # 计算均方根差（RMSE）
            if 'RMSE' in stats:
                sum_sq = sum(x**2 for x in all_numbers)
                results['RMSE'] = math.sqrt(sum_sq / n)

            return samples, results  # 返回（样本集，统计结果）

        return wrapper

    return decorator

# 用装饰器修饰作业2的生成函数
@stats_decorator(stats=['SUM', 'AVG', 'VAR', 'RMSE'])
def decorated_generate_samples(**kwargs):
    return generate_samples(**kwargs)

# 测试示例
if __name__ == "__main__":
    # 使用作业2的测试结构
    test_structure = {
        'type': 'list',
        'length': 3,
        'element': {
            'type': 'dict',
            'keys': {
                'id': {'type': 'int', 'min': 1, 'max': 1000},
                'scores': {
                    'type': 'tuple',
                    'elements': [
                        {'type': 'float', 'min': 0.0, 'max': 100.0},
                        {'type': 'float', 'min': 0.0, 'max': 100.0}
                    ]
                }
            }
        }
    }

    # 生成样本并获取统计结果
    samples, stats = decorated_generate_samples(structure=test_structure, samples=5)

    print("=== 生成的样本集 ===")
    for i, sample in enumerate(samples, 1):
        print(f"样本 {i}: {sample}")

    print("\n=== 统计结果 ===")
    for stat, value in stats.items():
        print(f"{stat}: {value:.4f}")
