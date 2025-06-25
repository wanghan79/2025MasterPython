import random
import string
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from functools import wraps


class StatsDecorator:
    """统计分析装饰器类"""

    def __init__(self, stats: List[str] = None):
        """
        初始化装饰器

        Args:
            stats: 需要计算的统计项列表，可选值：'SUM', 'AVG', 'VAR', 'RMSE'
        """
        self.stats = stats or ['SUM', 'AVG', 'VAR', 'RMSE']
        self.valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}

        # 验证统计项
        invalid_stats = set(self.stats) - self.valid_stats
        if invalid_stats:
            raise ValueError(f"Invalid statistics: {invalid_stats}")

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原始函数生成数据
            samples = func(*args, **kwargs)

            # 分析数据
            results = self.analyze(samples)

            # 将分析结果添加到返回数据中
            if isinstance(samples, list):
                return {
                    'samples': samples,
                    'statistics': results
                }
            return results

        return wrapper

    def analyze(self, data: Union[List, Dict]) -> Dict[str, Dict[str, float]]:
        """
        分析数据中的数值型数据

        Args:
            data: 要分析的数据

        Returns:
            统计结果字典
        """
        # 收集所有数值型数据
        numeric_values = self._collect_numeric_values(data)

        # 计算统计值
        results = {}
        for key, values in numeric_values.items():
            results[key] = self._calculate_statistics(values)

        return results

    def _collect_numeric_values(self, data: Any, prefix: str = '') -> Dict[str, List[float]]:
        """递归收集数值型数据"""
        numeric_values = {}

        if isinstance(data, (int, float)):
            numeric_values[prefix or 'value'] = [float(data)]
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                sub_values = self._collect_numeric_values(
                    item,
                    f"{prefix}[{i}]" if prefix else f"[{i}]"
                )
                for k, v in sub_values.items():
                    numeric_values[k] = v
        elif isinstance(data, dict):
            for k, v in data.items():
                sub_values = self._collect_numeric_values(
                    v,
                    f"{prefix}.{k}" if prefix else k
                )
                for sub_k, sub_v in sub_values.items():
                    numeric_values[sub_k] = sub_v

        return numeric_values

    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """计算统计值"""
        if not values:
            return {}

        results = {}
        n = len(values)

        if 'SUM' in self.stats:
            results['SUM'] = sum(values)

        if 'AVG' in self.stats:
            results['AVG'] = sum(values) / n

        if 'VAR' in self.stats or 'RMSE' in self.stats:
            mean = sum(values) / n
            squared_diff_sum = sum((x - mean) ** 2 for x in values)

            if 'VAR' in self.stats:
                results['VAR'] = squared_diff_sum / n

            if 'RMSE' in self.stats:
                results['RMSE'] = math.sqrt(squared_diff_sum / n)

        return results


class DataSampler:
    """随机数据生成器，支持生成各种嵌套结构的随机数据"""

    def __init__(self):
        # 定义基础数据类型的生成器
        self.type_generators = {
            'int': self._generate_int,
            'float': self._generate_float,
            'str': self._generate_str,
            'bool': self._generate_bool,
            'date': self._generate_date,
            'list': self._generate_list,
            'tuple': self._generate_tuple,
            'dict': self._generate_dict
        }

        # 默认配置
        self.default_config = {
            'int': {'min': 0, 'max': 100},
            'float': {'min': 0, 'max': 100, 'precision': 2},
            'str': {'min_length': 5, 'max_length': 10},
            'date': {'start_date': '2020-01-01', 'end_date': '2023-12-31'}
        }

    def _generate_int(self, **kwargs) -> int:
        """生成随机整数"""
        min_val = kwargs.get('min', self.default_config['int']['min'])
        max_val = kwargs.get('max', self.default_config['int']['max'])
        return random.randint(min_val, max_val)

    def _generate_float(self, **kwargs) -> float:
        """生成随机浮点数"""
        min_val = kwargs.get('min', self.default_config['float']['min'])
        max_val = kwargs.get('max', self.default_config['float']['max'])
        precision = kwargs.get('precision', self.default_config['float']['precision'])
        return round(random.uniform(min_val, max_val), precision)

    def _generate_str(self, **kwargs) -> str:
        """生成随机字符串"""
        min_length = kwargs.get('min_length', self.default_config['str']['min_length'])
        max_length = kwargs.get('max_length', self.default_config['str']['max_length'])
        length = random.randint(min_length, max_length)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def _generate_bool(self, **kwargs) -> bool:
        """生成随机布尔值"""
        return random.choice([True, False])

    def _generate_date(self, **kwargs) -> datetime:
        """生成随机日期"""
        start_date = datetime.strptime(
            kwargs.get('start_date', self.default_config['date']['start_date']),
            '%Y-%m-%d'
        )
        end_date = datetime.strptime(
            kwargs.get('end_date', self.default_config['date']['end_date']),
            '%Y-%m-%d'
        )
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        return start_date + timedelta(days=random_days)

    def _generate_list(self, **kwargs) -> List:
        """生成随机列表"""
        length = kwargs.get('length', random.randint(1, 5))
        item_type = kwargs.get('item_type', 'int')
        return [self.type_generators[item_type](**kwargs) for _ in range(length)]

    def _generate_tuple(self, **kwargs) -> Tuple:
        """生成随机元组"""
        return tuple(self._generate_list(**kwargs))

    def _generate_dict(self, **kwargs) -> Dict:
        """生成随机字典"""
        keys = kwargs.get('keys', [])
        if not keys:
            return {}

        result = {}
        for key, value_type in keys.items():
            if isinstance(value_type, dict):
                # 递归处理嵌套结构
                result[key] = self._generate_dict(**value_type)
            else:
                result[key] = self.type_generators[value_type](**kwargs)
        return result

    @StatsDecorator(['SUM', 'AVG', 'VAR', 'RMSE'])
    def generate_samples(self, structure: Dict, num_samples: int = 1, **kwargs) -> List[Dict]:
        """
        生成指定数量的随机样本

        Args:
            structure: 数据结构定义
            num_samples: 要生成的样本数量
            **kwargs: 其他配置参数

        Returns:
            生成的样本列表和统计结果
        """
        return [self._generate_dict(keys=structure, **kwargs) for _ in range(num_samples)]


def main():
    # 创建数据生成器实例
    sampler = DataSampler()

    # 定义数据结构
    user_structure = {
        'user_id': 'int',
        'username': 'str',
        'age': 'int',
        'height': 'float',
        'is_active': 'bool',
        'created_at': 'date',
        'scores': {
            'type': 'list',
            'item_type': 'float',
            'length': 3
        },
        'address': {
            'city': 'str',
            'zip_code': 'str',
            'coordinates': {
                'type': 'tuple',
                'item_type': 'float',
                'length': 2
            }
        }
    }

    # 生成5个样本并获取统计结果
    result = sampler.generate_samples(
        structure=user_structure,
        num_samples=5,
        int={'min': 1, 'max': 100},
        float={'min': 0, 'max': 10, 'precision': 2},
        str={'min_length': 3, 'max_length': 8}
    )

    # 打印生成的样本
    print("\n生成的随机样本：")
    for i, sample in enumerate(result['samples'], 1):
        print(f"\n样本 {i}:")
        for key, value in sample.items():
            print(f"{key}: {value}")

    # 打印统计结果
    print("\n统计结果：")
    for field, stats in result['statistics'].items():
        print(f"\n{field}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}")


if __name__ == "__main__":
    main()