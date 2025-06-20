import random
import string
import math
from typing import Any, Dict, List, Tuple, Union
from functools import wraps


def statistical_analysis(*stats):
    """
    带参数的统计分析修饰器

    参数:
    *stats: 需要计算的统计项，支持 'SUM', 'AVG', 'VAR', 'RMSE'

    使用示例:
    @statistical_analysis('SUM', 'AVG')
    @statistical_analysis('SUM', 'AVG', 'VAR', 'RMSE')
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原函数获取样本数据
            samples = func(*args, **kwargs)

            # 提取所有数值型数据
            numeric_values = extract_numeric_values(samples)

            if not numeric_values:
                print("警告: 样本中没有找到数值型数据")
                return samples, {}

            # 计算统计项
            results = {}

            if 'SUM' in stats:
                results['SUM'] = calculate_sum(numeric_values)

            if 'AVG' in stats:
                results['AVG'] = calculate_average(numeric_values)

            if 'VAR' in stats:
                results['VAR'] = calculate_variance(numeric_values)

            if 'RMSE' in stats:
                results['RMSE'] = calculate_rmse(numeric_values)

            # 输出统计信息
            print_statistics(numeric_values, results, stats)

            return samples, results

        return wrapper

    return decorator


def extract_numeric_values(data):
    """递归提取嵌套数据结构中的所有数值型数据"""
    numeric_values = []

    def recursive_extract(obj):
        if isinstance(obj, (int, float)):
            numeric_values.append(float(obj))
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                recursive_extract(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                recursive_extract(value)
        # 忽略其他类型（字符串、布尔值等）

    # 处理单个样本或样本列表
    if isinstance(data, list) and len(data) > 0:
        # 检查是否为样本列表
        for sample in data:
            recursive_extract(sample)
    else:
        recursive_extract(data)

    return numeric_values


def calculate_sum(values):
    """计算求和"""
    return sum(values)


def calculate_average(values):
    """计算均值"""
    if not values:
        return 0
    return sum(values) / len(values)


def calculate_variance(values):
    """计算方差"""
    if len(values) < 2:
        return 0

    avg = calculate_average(values)
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    return variance


def calculate_rmse(values):
    """计算均方根差（相对于均值）"""
    if len(values) < 2:
        return 0

    avg = calculate_average(values)
    mse = sum((x - avg) ** 2 for x in values) / len(values)
    return math.sqrt(mse)


def print_statistics(values, results, requested_stats):
    """打印统计结果"""
    print("\n" + "=" * 50)
    print("数值统计分析结果")
    print("=" * 50)
    print(f"数值总数: {len(values)}")
    print(f"数值范围: [{min(values):.2f}, {max(values):.2f}]")
    print("-" * 30)

    for stat in requested_stats:
        if stat in results:
            print(f"{stat}: {results[stat]:.4f}")

    print("=" * 50)


# 原始的样本生成函数（从作业2复制过来并稍作修改）
def generate_random_samples(**kwargs):
    """
    构造任意嵌套数据类型随机样本生成函数
    """
    count = kwargs.get('count', 1)
    structure = kwargs.get('structure', {'type': 'int', 'min': 1, 'max': 10})

    def generate_single_value(spec):
        if not isinstance(spec, dict) or 'type' not in spec:
            return spec

        data_type = spec['type']

        if data_type == 'int':
            min_val = spec.get('min', 0)
            max_val = spec.get('max', 100)
            return random.randint(min_val, max_val)

        elif data_type == 'float':
            min_val = spec.get('min', 0.0)
            max_val = spec.get('max', 1.0)
            return round(random.uniform(min_val, max_val), 2)

        elif data_type == 'str':
            length = spec.get('length', 5)
            charset = spec.get('charset', 'letters')

            if charset == 'letters':
                chars = string.ascii_letters
            elif charset == 'digits':
                chars = string.digits
            elif charset == 'alphanumeric':
                chars = string.ascii_letters + string.digits
            else:
                chars = charset

            return ''.join(random.choice(chars) for _ in range(length))

        elif data_type == 'bool':
            return random.choice([True, False])

        elif data_type == 'list':
            length = spec.get('length', 3)
            element_spec = spec.get('element', {'type': 'int'})
            return [generate_single_value(element_spec) for _ in range(length)]

        elif data_type == 'tuple':
            length = spec.get('length', 3)
            element_spec = spec.get('element', {'type': 'int'})
            return tuple(generate_single_value(element_spec) for _ in range(length))

        elif data_type == 'dict':
            keys = spec.get('keys', ['key1', 'key2', 'key3'])
            value_spec = spec.get('values', {'type': 'str'})

            result = {}
            for key in keys:
                if isinstance(key, dict):
                    actual_key = generate_single_value(key)
                else:
                    actual_key = key
                result[actual_key] = generate_single_value(value_spec)
            return result

        elif data_type == 'set':
            length = spec.get('length', 3)
            element_spec = spec.get('element', {'type': 'int'})
            elements = []
            attempts = 0
            while len(set(elements)) < length and attempts < length * 10:
                elements.append(generate_single_value(element_spec))
                attempts += 1
            return set(elements[:length])

        elif data_type == 'mixed':
            options = spec.get('options', [{'type': 'int'}, {'type': 'str'}])
            chosen_spec = random.choice(options)
            return generate_single_value(chosen_spec)

        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

    samples = []
    for i in range(count):
        sample = generate_single_value(structure)
        samples.append(sample)

    return samples if count > 1 else samples[0]


# 类修饰器版本
class StatisticalAnalyzer:
    """类修饰器版本的统计分析器"""

    def __init__(self, *stats):
        self.stats = stats

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            numeric_values = extract_numeric_values(samples)

            if not numeric_values:
                print("警告: 样本中没有找到数值型数据")
                return samples, {}

            results = {}

            if 'SUM' in self.stats:
                results['SUM'] = calculate_sum(numeric_values)

            if 'AVG' in self.stats:
                results['AVG'] = calculate_average(numeric_values)

            if 'VAR' in self.stats:
                results['VAR'] = calculate_variance(numeric_values)

            if 'RMSE' in self.stats:
                results['RMSE'] = calculate_rmse(numeric_values)

            print_statistics(numeric_values, results, self.stats)
            return samples, results

        return wrapper


def demo_statistical_decorators():
    """演示统计修饰器的使用"""
    print("数值统计修饰器演示")
    print("=" * 60)

    # 示例1: 使用函数修饰器 - 计算所有统计项
    print("\n【示例1: 完整统计分析】")

    @statistical_analysis('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_numeric_samples():
        return generate_random_samples(
            count=5,
            structure={
                'type': 'list',
                'length': 4,
                'element': {'type': 'float', 'min': 1.0, 'max': 10.0}
            }
        )

    samples1, stats1 = generate_numeric_samples()
    print(f"样本数据: {samples1}")

    # 示例2: 使用函数修饰器 - 只计算部分统计项
    print("\n【示例2: 部分统计分析】")

    @statistical_analysis('SUM', 'AVG')
    def generate_mixed_samples():
        return generate_random_samples(
            count=3,
            structure={
                'type': 'dict',
                'keys': ['numbers', 'scores', 'info'],
                'values': {
                    'type': 'mixed',
                    'options': [
                        {
                            'type': 'list',
                            'length': 3,
                            'element': {'type': 'int', 'min': 50, 'max': 100}
                        },
                        {'type': 'float', 'min': 0.0, 'max': 5.0},
                        {'type': 'str', 'length': 5}
                    ]
                }
            }
        )

    samples2, stats2 = generate_mixed_samples()
    print(f"样本数据: {samples2}")

    # 示例3: 使用类修饰器
    print("\n【示例3: 类修饰器版本】")

    @StatisticalAnalyzer('AVG', 'VAR', 'RMSE')
    def generate_nested_samples():
        return generate_random_samples(
            count=4,
            structure={
                'type': 'tuple',
                'length': 2,
                'element': {
                    'type': 'mixed',
                    'options': [
                        {'type': 'int', 'min': 1, 'max': 20},
                        {
                            'type': 'list',
                            'length': 3,
                            'element': {'type': 'float', 'min': -2.0, 'max': 2.0}
                        }
                    ]
                }
            }
        )

    samples3, stats3 = generate_nested_samples()
    print(f"样本数据: {samples3}")

    # 示例4: 学生成绩统计实例
    print("\n【示例4: 实际应用 - 学生成绩统计】")

    @statistical_analysis('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_student_grades():
        return generate_random_samples(
            count=10,
            structure={
                'type': 'dict',
                'keys': ['student_id', 'math', 'english', 'science'],
                'values': {
                    'type': 'mixed',
                    'options': [
                        {'type': 'int', 'min': 20210001, 'max': 20210010},
                        {'type': 'float', 'min': 60.0, 'max': 100.0},
                        {'type': 'float', 'min': 60.0, 'max': 100.0},
                        {'type': 'float', 'min': 60.0, 'max': 100.0}
                    ]
                }
            }
        )

    students, grade_stats = generate_student_grades()
    print("学生成绩样本:")
    for i, student in enumerate(students[:3], 1):  # 只显示前3个
        print(f"  学生{i}: {student}")
    if len(students) > 3:
        print(f"  ... 还有{len(students) - 3}个样本")

    # 输出所有统计结果汇总
    print("\n" + "=" * 60)
    print("所有测试的统计结果汇总")
    print("=" * 60)
    print(f"示例1统计结果: {stats1}")
    print(f"示例2统计结果: {stats2}")
    print(f"示例3统计结果: {stats3}")
    print(f"示例4统计结果: {grade_stats}")


if __name__ == "__main__":
    # 设置随机种子保证结果可重现
    random.seed(42)
    demo_statistical_decorators()