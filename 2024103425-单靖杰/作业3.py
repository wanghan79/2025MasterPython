import random
import string
import datetime
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import numpy as np


def extract_numerical_values(data: Any) -> List[Union[int, float]]:
    """递归提取数据结构中所有数值型叶节点的值"""
    result = []

    if isinstance(data, (int, float)):
        result.append(data)
    elif isinstance(data, list) or isinstance(data, tuple):
        for item in data:
            result.extend(extract_numerical_values(item))
    elif isinstance(data, dict):
        for value in data.values():
            result.extend(extract_numerical_values(value))

    return result


def stats_analysis(include: List[str] = ['SUM', 'AVG', 'VAR', 'RMSE']):
    """
    统计分析装饰器，计算数据中数值型叶节点的统计特征

    Args:
        include: 要计算的统计项列表，可选值为'SUM', 'AVG', 'VAR', 'RMSE'
    """
    valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
    invalid_stats = [stat for stat in include if stat not in valid_stats]

    if invalid_stats:
        raise ValueError(f"Invalid statistics specified: {invalid_stats}. Valid options are {list(valid_stats)}.")

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # 调用原始函数生成数据
            result = func(*args, **kwargs)

            # 如果结果是元组，假设第二个元素是已有统计信息
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                samples, existing_stats = result
            else:
                samples = result
                existing_stats = {}

            # 提取所有数值型叶节点
            numerical_values = extract_numerical_values(samples)

            # 计算统计特征
            stats = {}
            if numerical_values:
                if 'SUM' in include:
                    stats['SUM'] = sum(numerical_values)
                if 'AVG' in include:
                    stats['AVG'] = np.mean(numerical_values)
                if 'VAR' in include:
                    stats['VAR'] = np.var(numerical_values)
                if 'RMSE' in include:
                    stats['RMSE'] = np.sqrt(np.mean(np.square(numerical_values)))

            # 合并现有统计信息
            combined_stats = {**existing_stats, **stats}

            return samples, combined_stats

        return wrapper

    return decorator


def analyze(cls):
    """
    类装饰器，为DataSampler类添加统计分析功能

    添加了一个analyze方法，可对生成的数据进行统计分析
    """
    original_generate_samples = cls.generate_samples

    def new_generate_samples(self, num_samples: int, **kwargs):
        samples = original_generate_samples(self, num_samples, **kwargs)
        return samples

    @stats_analysis()
    def analyze_method(self, samples):
        return samples

    cls.generate_samples = new_generate_samples
    cls.analyze = analyze_method

    return cls


@analyze
class DataSampler:

    def __init__(self, seed: Optional[int] = None):
        """
        初始化数据生成器，可设置随机数种子以保证结果可复现。

        Args:
            seed: 随机数种子
        """
        if seed is not None:
            random.seed(seed)

        # 注册各数据类型的生成函数
        self.type_generators = {
            int: self._generate_int,
            float: self._generate_float,
            str: self._generate_str,
            list: self._generate_list,
            tuple: self._generate_tuple,
            dict: self._generate_dict
        }

    def generate_samples(self, num_samples: int, **kwargs) -> List[Any]:
        """
        生成指定数量的随机样本

        Args:
            num_samples: 要生成的样本数量
            **kwargs: 样本的数据结构配置

        Returns:
            生成的样本列表
        """
        return [self._generate_sample(kwargs) for _ in range(num_samples)]

    def _generate_sample(self, config: Dict) -> Any:
        """
        根据配置生成单个样本

        Args:
            config: 样本配置

        Returns:
            生成的样本数据
        """
        data_type = config.get('type')

        if data_type not in self.type_generators:
            raise ValueError(f"Unsupported data type: {data_type}")

        return self.type_generators[data_type](config)

    def _generate_int(self, config: Dict) -> int:
        """生成随机整数"""
        min_val = config.get('min', 0)
        max_val = config.get('max', 100)
        return random.randint(min_val, max_val)

    def _generate_float(self, config: Dict) -> float:
        """生成随机浮点数"""
        min_val = config.get('min', 0.0)
        max_val = config.get('max', 1.0)
        precision = config.get('precision', 2)
        return round(random.uniform(min_val, max_val), precision)

    def _generate_str(self, config: Dict) -> str:
        """生成随机字符串"""
        length = config.get('length', 10)
        charset = config.get('charset', string.ascii_letters + string.digits)
        return ''.join(random.choices(charset, k=length))

    def _generate_list(self, config: Dict) -> List:
        """生成随机列表"""
        min_length = config.get('min_length', 1)
        max_length = config.get('max_length', 10)
        length = random.randint(min_length, max_length)

        element_config = config.get('element_config', {})
        return [self._generate_sample(element_config) for _ in range(length)]

    def _generate_tuple(self, config: Dict) -> Tuple:
        """生成随机元组"""
        elements_config = config.get('elements_config', [{}])
        return tuple(self._generate_sample(elem_config) for elem_config in elements_config)

    def _generate_dict(self, config: Dict) -> Dict:
        """生成随机字典"""
        fields = config.get('fields', {})
        return {key: self._generate_sample(field_config) for key, field_config in fields.items()}


# 使用示例
if __name__ == "__main__":
    # 创建数据生成器实例
    sampler = DataSampler(seed=42)

    # 定义用户数据结构配置
    user_config = {
        'type': dict,
        'fields': {
            'id': {'type': int, 'min': 1, 'max': 1000},
            'name': {'type': str, 'length': 8},
            'age': {'type': int, 'min': 18, 'max': 90},
            'is_active': {'type': bool, 'true_prob': 0.8},
            'height': {'type': float, 'min': 1.4, 'max': 2.1, 'precision': 2},
            'registration_date': {'type': datetime.date, 'start_date': datetime.date(2020, 1, 1)},
            'hobbies': {
                'type': list,
                'min_length': 2,
                'max_length': 5,
                'element_config': {'type': str, 'length': 6}
            },
            'address': {
                'type': dict,
                'fields': {
                    'street': {'type': str, 'length': 15},
                    'city': {'type': str, 'length': 10},
                    'zip_code': {'type': str, 'length': 5, 'charset': string.digits},
                    'apartment_number': {'type': int, 'min': 1, 'max': 200}
                }
            }
        }
    }

    @stats_analysis(include=['AVG', 'RMSE'])
    def generate_custom_data():
        return [
            {'a': 1, 'b': 2.5},
            {'a': 3, 'b': 4.5},
            {'a': 5, 'b': 6.5}
        ]


    custom_samples, custom_stats = generate_custom_data()
    print("\n自定义数据生成器统计结果:")
    for stat, value in custom_stats.items():
        print(f"{stat}: {value}")