import random
import string
from datetime import datetime, timedelta
import numpy as np
from functools import wraps


def generate_random_value(data_type, **kwargs):
    """生成单个随机值，支持多种数据类型"""
    if data_type == int:
        min_val = kwargs.get('min', 0)
        max_val = kwargs.get('max', 100)
        return random.randint(min_val, max_val)

    elif data_type == float:
        min_val = kwargs.get('min', 0.0)
        max_val = kwargs.get('max', 1.0)
        precision = kwargs.get('precision', 2)
        return round(random.uniform(min_val, max_val), precision)

    elif data_type == str:
        length = kwargs.get('length', 10)
        charset = kwargs.get('charset', string.ascii_letters + string.digits)
        return ''.join(random.choice(charset) for _ in range(length))

    elif data_type == bool:
        return random.choice([True, False])

    elif data_type == datetime:
        start_date = kwargs.get('start_date', datetime(2000, 1, 1))
        end_date = kwargs.get('end_date', datetime.now())
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        return start_date + timedelta(days=random_days)

    else:
        raise ValueError(f"不支持的数据类型: {data_type}")


def generate_nested_structure(structure, **kwargs):
    """根据指定的嵌套结构生成随机数据"""
    if isinstance(structure, dict):
        result = {}
        for key, value in structure.items():
            if callable(value):
                # 如果值是一个类型，直接生成该类型的随机值
                result[key] = generate_random_value(value, **kwargs)
            elif isinstance(value, dict) or isinstance(value, list) or isinstance(value, tuple):
                # 如果值是嵌套结构，递归生成
                result[key] = generate_nested_structure(value, **kwargs)
            else:
                # 其他情况保持原值
                result[key] = value
        return result

    elif isinstance(structure, list):
        if not structure:  # 空列表
            return []
        # 假设列表中的元素类型都相同，使用第一个元素作为结构模板
        element_structure = structure[0]
        min_len = kwargs.get('min_length', 1)
        max_len = kwargs.get('max_length', 5)
        length = random.randint(min_len, max_len)
        return [generate_nested_structure(element_structure, **kwargs) for _ in range(length)]

    elif isinstance(structure, tuple):
        if not structure:  # 空元组
            return ()
        # 元组中的每个元素可以是不同类型
        return tuple(generate_nested_structure(item, **kwargs) for item in structure)

    else:
        # 基本数据类型
        return generate_random_value(structure, **kwargs)


def stats_decorator(statistics=None):
    """
    统计装饰器，用于计算数据样本的统计特征

    参数:statistics: 需要计算的统计项列表，可选值为 'SUM', 'AVG', 'VAR', 'RMSE'
    """
    if statistics is None:
        statistics = ['SUM', 'AVG', 'VAR', 'RMSE']

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原始函数生成样本数据
            samples = func(*args, **kwargs)

            # 分析数据并计算统计结果
            results = analyze(samples, statistics)

            # 返回原始样本和统计结果
            return {
                'samples': samples,
                'statistics': results
            }

        return wrapper

    return decorator


def analyze(samples, statistics):
    """
    分析样本数据中的数值型叶节点并计算指定的统计特征

    参数:samples: 样本数据列表 statistics: 需要计算的统计项列表

    返回统计结果字典
    """
    # 收集所有数值型叶节点的值
    numerical_values = {}

    def extract_numerical_values(path, value):
        """递归提取数值型叶节点的值"""
        if isinstance(value, (int, float)):
            # 数值型叶节点，记录路径和值
            numerical_values.setdefault(path, []).append(value)
        elif isinstance(value, dict):
            # 字典类型，递归处理每个键值对
            for key, val in value.items():
                new_path = f"{path}.{key}" if path else key
                extract_numerical_values(new_path, val)
        elif isinstance(value, list) or isinstance(value, tuple):
            # 列表或元组类型，递归处理每个元素
            for i, item in enumerate(value):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                extract_numerical_values(new_path, item)

    # 对每个样本提取数值型叶节点
    for i, sample in enumerate(samples):
        extract_numerical_values(f"sample[{i}]", sample)

    # 计算统计结果
    results = {}
    for path, values in numerical_values.items():
        path_stats = {}

        if 'SUM' in statistics:
            path_stats['SUM'] = sum(values)

        if 'AVG' in statistics:
            path_stats['AVG'] = np.mean(values)

        if 'VAR' in statistics:
            path_stats['VAR'] = np.var(values)

        if 'RMSE' in statistics:
            path_stats['RMSE'] = np.sqrt(np.mean(np.square(values)))

        results[path] = path_stats

    return results


# 使用装饰器修饰DataSampler函数
@stats_decorator(statistics=['SUM', 'AVG', 'VAR', 'RMSE'])
def DataSampler(structure, num_samples=1, **kwargs):
    """
    生成结构化的随机测试数据样本集

    参数: structure: 数据结构模板，可以是字典、列表、元组或基本数据类型
    num_samples: 要生成的样本数量
    **kwargs: 其他可选参数，如数值范围、字符串长度等

    返回:
    - 包含样本数据和统计结果的字典
    """
    return [generate_nested_structure(structure, **kwargs) for _ in range(num_samples)]


# 示例用法
if __name__ == "__main__":
    # 定义用户数据结构模板
    user_structure = {
        "id": int,
        "name": str,
        "age": int,
        "is_active": bool,
        "height": float,
        "registration_date": datetime,
        "hobbies": [str],
        "address": {
            "street": str,
            "city": str,
            "zip_code": str,
            "coordinates": (float, float)
        }
    }

    # 生成5个用户样本并进行统计分析
    result = DataSampler(
        user_structure,
        num_samples=5,
        min=0,  # 数值最小值
        max=100,  # 数值最大值
        length=8,  # 字符串长度
        min_length=2  # 列表最小长度
    )

    # 打印生成的样本
    print("生成的样本数据:")
    for i, user in enumerate(result['samples'], 1):
        print(f"用户样本 {i}:")
        for key, value in user.items():
            print(f"  {key}: {value}")
        print()

    # 打印统计结果
    print("\n统计分析结果:")
    for path, stats in result['statistics'].items():
        print(f"路径: {path}")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.4f}")
        print()
