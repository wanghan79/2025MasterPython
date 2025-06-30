import random
import string
from datetime import datetime, timedelta
import numpy as np


class DataSampler:
    """
    随机数据生成器，支持动态生成嵌套的字典、列表、元组等数据结构
    """

    def __init__(self):
        # 注册基本数据类型的生成函数
        self.generators = {
            'int': self._generate_int,
            'float': self._generate_float,
            'str': self._generate_str,
            'bool': self._generate_bool,
            'date': self._generate_date
        }

    def generate_samples(self, **kwargs):
        """
        生成样本集，样本的嵌套数据结构及样本个数均由kwargs参数给定

        参数:
            - kwargs: 包含数据结构定义和样本数量的参数
                - samples: 要生成的样本数量
                - structure: 定义数据结构的字典

        返回:
            生成的样本集列表
        """
        samples = kwargs.get('samples', 1)
        structure = kwargs.get('structure', {})

        return [self._generate_sample(structure) for _ in range(samples)]

    def _generate_sample(self, structure):
        """根据结构定义生成单个样本"""
        if isinstance(structure, dict):
            # 处理字典类型
            if '_type' in structure:
                # 特殊类型定义
                type_name = structure['_type']
                if type_name in self.generators:
                    return self.generators[type_name](structure)
                elif type_name == 'list':
                    return self._generate_list(structure)
                elif type_name == 'tuple':
                    return self._generate_tuple(structure)
                elif type_name == 'dict':
                    return self._generate_dict(structure)
            else:
                # 普通字典
                return {k: self._generate_sample(v) for k, v in structure.items()}

        elif isinstance(structure, list):
            # 列表中的每个元素代表不同的可能性
            if structure:
                choice = random.choice(structure)
                return self._generate_sample(choice)

        return structure  # 基本类型值直接返回

    def _generate_int(self, params):
        """生成随机整数"""
        min_val = params.get('min', 0)
        max_val = params.get('max', 100)
        return random.randint(min_val, max_val)

    def _generate_float(self, params):
        """生成随机浮点数"""
        min_val = params.get('min', 0.0)
        max_val = params.get('max', 1.0)
        precision = params.get('precision', 2)
        return round(random.uniform(min_val, max_val), precision)

    def _generate_str(self, params):
        """生成随机字符串"""
        length = params.get('length', 10)
        charset = params.get('charset', string.ascii_letters + string.digits)
        return ''.join(random.choice(charset) for _ in range(length))

    def _generate_bool(self, params):
        """生成随机布尔值"""
        return random.choice([True, False])

    def _generate_date(self, params):
        """生成随机日期"""
        start_date = params.get('start', datetime(2000, 1, 1))
        end_date = params.get('end', datetime.now())

        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        return (start_date + timedelta(days=random_days)).date()

    def _generate_list(self, params):
        """生成随机列表"""
        min_len = params.get('min_length', 0)
        max_len = params.get('max_length', 10)
        length = random.randint(min_len, max_len)

        element_type = params.get('element_type', {'_type': 'int'})
        return [self._generate_sample(element_type) for _ in range(length)]

    def _generate_tuple(self, params):
        """生成随机元组"""
        elements = params.get('elements', [{'_type': 'int'}])
        return tuple(self._generate_sample(e) for e in elements)

    def _generate_dict(self, params):
        """生成随机字典"""
        fields = params.get('fields', {})
        return {k: self._generate_sample(v) for k, v in fields.items()}


def stats_decorator(*stats):
    """
    统计装饰器，用于计算数据样本集中所有数值型数据的统计特征

    参数:
        stats: 要计算的统计项，可选值为 'SUM', 'AVG', 'VAR', 'RMSE'
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 调用原始函数生成样本
            samples = func(*args, **kwargs)

            # 收集所有数值型叶节点数据
            numerical_data = _collect_numerical_data(samples)

            # 计算统计结果
            results = _calculate_statistics(numerical_data, stats)

            # 包装原始结果和统计结果
            return {
                'samples': samples,
                'statistics': results
            }

        return wrapper

    return decorator


def _collect_numerical_data(samples):
    """收集样本集中所有数值型叶节点数据"""
    numerical_data = []

    def traverse(obj):
        if isinstance(obj, (int, float)):
            numerical_data.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                traverse(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                traverse(item)

    for sample in samples:
        traverse(sample)

    return numerical_data


def _calculate_statistics(data, stats):
    """计算指定的统计项"""
    results = {}

    if not data:
        return results

    if 'SUM' in stats:
        results['SUM'] = sum(data)

    if 'AVG' in stats:
        results['AVG'] = sum(data) / len(data)

    if 'VAR' in stats:
        avg = sum(data) / len(data)
        results['VAR'] = sum((x - avg) ** 2 for x in data) / len(data)

    if 'RMSE' in stats:
        avg = sum(data) / len(data)
        mse = sum((x - avg) ** 2 for x in data) / len(data)
        results['RMSE'] = np.sqrt(mse)

    return results


# 使用示例
if __name__ == "__main__":
    sampler = DataSampler()


    # 使用装饰器分析用户数据
    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_and_analyze_users():
        user_structure = {
            'id': {'_type': 'int', 'min': 1, 'max': 1000},
            'name': {'_type': 'str', 'length': 8},
            'age': {'_type': 'int', 'min': 18, 'max': 80},
            'is_active': {'_type': 'bool'},
            'created_at': {'_type': 'date', 'start': datetime(2020, 1, 1)},
            'interests': {
                '_type': 'list',
                'element_type': {'_type': 'str', 'length': 5}
            },
            'address': {
                'street': {'_type': 'str', 'length': 15},
                'city': {'_type': 'str', 'length': 10},
                'zip': {'_type': 'str', 'length': 5, 'charset': string.digits}
            }
        }
        return sampler.generate_samples(samples=3, structure=user_structure)


    result = generate_and_analyze_users()
    print("用户数据样本:")
    for user in result['samples']:
        print(user)

    print("\n统计结果:")
    for stat, value in result['statistics'].items():
        print(f"{stat}: {value}")


    # 使用装饰器分析复杂数据
    @stats_decorator('SUM', 'AVG')
    def generate_and_analyze_complex_data():
        complex_structure = {
            'items': {
                '_type': 'list',
                'min_length': 2,
                'max_length': 5,
                'element_type': {
                    '_type': 'dict',
                    'fields': {
                        'item_id': {'_type': 'int', 'min': 100, 'max': 200},
                        'price': {'_type': 'float', 'min': 0.01, 'max': 99.99, 'precision': 2},
                        'tags': {
                            '_type': 'tuple',
                            'elements': [
                                {'_type': 'str', 'length': 4},
                                {'_type': 'str', 'length': 6}
                            ]
                        }
                    }
                }
            },
            'metadata': {
                'timestamp': {'_type': 'date', 'start': datetime(2023, 1, 1)},
                'version': {'_type': 'str', 'length': 5, 'charset': string.digits + '.'}
            }
        }
        return sampler.generate_samples(samples=2, structure=complex_structure)


    complex_result = generate_and_analyze_complex_data()
    print("\n复杂数据样本:")
    for data in complex_result['samples']:
        print(data)

    print("\n统计结果:")
    for stat, value in complex_result['statistics'].items():
        print(f"{stat}: {value}")    