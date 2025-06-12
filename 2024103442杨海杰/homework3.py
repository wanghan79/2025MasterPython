import random
import datetime
import math
import string
from collections.abc import Callable
from typing import Any, Dict, List, Tuple


class DataSampler:
    """优化的结构化数据生成器"""

    TYPE_GENERATORS = {
        'int': lambda: random.randint(1, 100),
        'float': lambda: round(random.uniform(0.1, 10.0), 4),
        'str': lambda: ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 8))),
        'bool': lambda: random.choice([True, False]),
        'date': lambda: datetime.datetime.today() - datetime.timedelta(days=random.randint(0, 365 * 2)),
    }

    def generate(self, **kwargs) -> List[Dict]:
        """生成随机样本集"""
        structure = kwargs.get('structure', {})
        sample_count = kwargs.get('sample_count', 1)
        return [self._create_sample(structure) for _ in range(sample_count)]

    def _create_sample(self, spec: Any) -> Any:
        """递归创建单个样本"""
        if isinstance(spec, dict):
            return {key: self._create_sample(value) for key, value in spec.items()}

        if isinstance(spec, list):
            return [self._create_sample(item) for item in spec]

        if isinstance(spec, tuple):
            return tuple(self._create_sample(item) for item in spec)

        if callable(spec):
            return self._handle_callable(spec)

        if isinstance(spec, str) and spec in self.TYPE_GENERATORS:
            return self.TYPE_GENERATORS[spec]()

        return spec

    def _handle_callable(self, spec) -> Any:
        """处理可调用生成器函数"""
        try:
            if spec == random.choice:
                choices = self._random_elements()
                return spec(choices)
            elif spec == random.randint:
                return spec(1, 100)
            elif spec == random.uniform:
                return spec(0.1, 10.0)
            else:
                return spec()
        except Exception:
            return None

    def _random_elements(self):
        """生成随机元素集合"""
        return [
            ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 6)))
            for _ in range(random.randint(2, 4))
        ]


def stats_decorator(*metrics):

    valid_metrics = {'sum', 'avg', 'var', 'rmse'}
    stats_to_calc = {m.lower() for m in metrics if m.lower() in valid_metrics}

    if not stats_to_calc:
        raise ValueError("必须指定至少一个有效的统计指标: sum, avg, var, rmse")

    def extract_numbers(data, nums):
        """递归提取数值型数据"""
        if isinstance(data, (int, float)):
            nums.append(data)
        elif isinstance(data, (list, tuple)):
            for item in data:
                extract_numbers(item, nums)
        elif isinstance(data, dict):
            for value in data.values():
                extract_numbers(value, nums)

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成产品样本
            samples = func(*args, **kwargs)

            results = []
            for i, sample in enumerate(samples, 1):
                # 提取当前样本的所有数值
                all_nums = []
                extract_numbers(sample, all_nums)

                # 计算统计指标
                stats = {}
                if all_nums:
                    if 'sum' in stats_to_calc:
                        stats['sum'] = sum(all_nums)

                    mean = sum(all_nums) / len(all_nums)

                    if 'avg' in stats_to_calc:
                        stats['mean'] = mean

                    if 'var' in stats_to_calc or 'rmse' in stats_to_calc:
                        variance = sum((x - mean) ** 2 for x in all_nums) / len(all_nums)

                        if 'var' in stats_to_calc:
                            stats['variance'] = variance

                        if 'rmse' in stats_to_calc:
                            stats['rmse'] = math.sqrt(variance)

                # 添加到结果集
                results.append({
                    "index": i,
                    "sample": sample,
                    "stats": stats
                })

            return results

        return wrapper

    return decorator


# 使用示例
if __name__ == "__main__":
    # 定义产品数据结构
    product_structure = {
        "product_name": 'str',
        "quantity": 'int',
        "weight": 'float',
        "attributes": [
            {"colors": lambda: [''.join(random.choices(string.ascii_uppercase, k=4))
                                for _ in range(2)]},
            {"sizes": lambda: tuple(random.randint(30, 50) for _ in range(3))},
            {"ratings": lambda: [round(random.uniform(1.0, 5.0), 4) for _ in range(5)]},
            {"labels": lambda: tuple(''.join(random.choices(string.ascii_uppercase, k=3))
                                     for _ in range(2))}
        ],
        "category": random.choice,
        "in_stock": 'bool',
        "added_date": 'date'
    }

    # 创建数据采样器
    sampler = DataSampler()


    # 应用装饰器生成带统计的数据
    @stats_decorator('avg', 'var', 'rmse', 'sum')
    def generate_product_samples(count=3):
        return sampler.generate(
            structure=product_structure,
            sample_count=count
        )


    # 生成样本并统计
    products_with_stats = generate_product_samples()

    # 打印结果
    for product in products_with_stats:
        print(f"\n--- 产品样本 {product['index']} ---")
        print("原始数据:")

        # 格式化日期以便打印
        formatted_data = product['sample'].copy()
        if 'added_date' in formatted_data and isinstance(formatted_data['added_date'], datetime.datetime):
            formatted_data['added_date'] = formatted_data['added_date'].strftime("%Y-%m-%d")

        # 打印样本数据
        print(formatted_data)

        # 打印统计结果
        if product['stats']:
            print("该样本的统计结果 :")
            print(product['stats'])
        else:
            print("该样本没有找到可统计的数值数据")