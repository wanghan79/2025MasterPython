import numpy as np
from functools import wraps
from datetime import datetime, timedelta
import random
import string

def generate_random_value(data_type, **kwargs):
    """根据指定的数据类型生成随机值"""
    if data_type == int:
        return random.randint(kwargs.get('min', 0), kwargs.get('max', 100))
    elif data_type == float:
        return random.uniform(kwargs.get('min', 0.0), kwargs.get('max', 1.0))
    elif data_type == str:
        length = kwargs.get('length', 10)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    elif data_type == bool:
        return random.choice([True, False])
    elif data_type == datetime:
        start_date = kwargs.get('start_date', datetime(2000, 1, 1))
        end_date = kwargs.get('end_date', datetime.now())
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        return start_date + timedelta(days=random_days)
    else:
        return None


def generate_nested_structure(structure, **kwargs):
    """根据指定的结构生成嵌套数据"""
    if isinstance(structure, type):
        return generate_random_value(structure, **kwargs)

    if isinstance(structure, list):
        if not structure:  # 空列表
            return []
        element_structure = structure[0]  # 假设列表中所有元素结构相同
        min_len = kwargs.get('min_len', 1)
        max_len = kwargs.get('max_len', 5)
        length = random.randint(min_len, max_len)
        return [generate_nested_structure(element_structure, **kwargs) for _ in range(length)]

    if isinstance(structure, tuple):
        if not structure:  # 空元组
            return ()
        return tuple(generate_nested_structure(item, **kwargs) for item in structure)

    if isinstance(structure, dict):
        if 'type' in structure:  # 特殊情况：指定类型的字典
            data_type = structure['type']
            return generate_random_value(data_type, **structure)
        return {
            key: generate_nested_structure(value, **kwargs)
            for key, value in structure.items()
        }

    return None

def stats_decorator(*stats):
    """
    统计装饰器，用于计算生成数据中的数值型字段的统计特征

    参数:
        *stats: 要计算的统计项，可选值为 'SUM', 'AVG', 'VAR', 'RMSE'
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原始函数生成数据
            samples = func(*args, **kwargs)

            # 收集所有数值型叶节点数据
            numerical_data = {}

            def collect_numerical(node, path=""):
                if isinstance(node, (int, float, np.number)):
                    if path not in numerical_data:
                        numerical_data[path] = []
                    numerical_data[path].append(node)
                elif isinstance(node, list):
                    for i, item in enumerate(node):
                        collect_numerical(item, f"{path}[{i}]" if path else f"[{i}]")
                elif isinstance(node, dict):
                    for key, value in node.items():
                        collect_numerical(value, f"{path}.{key}" if path else key)
                elif isinstance(node, tuple):
                    for i, item in enumerate(node):
                        collect_numerical(item, f"{path}[{i}]" if path else f"[{i}]")

            # 遍历所有样本收集数值数据
            for i, sample in enumerate(samples):
                collect_numerical(sample, f"sample_{i}")

            # 计算统计结果
            result = {}
            for path, values in numerical_data.items():
                path_stats = {}
                if 'SUM' in stats:
                    path_stats['SUM'] = sum(values)
                if 'AVG' in stats:
                    path_stats['AVG'] = np.mean(values)
                if 'VAR' in stats:
                    path_stats['VAR'] = np.var(values)
                if 'RMSE' in stats:
                    path_stats['RMSE'] = np.sqrt(np.mean(np.square(values)))
                result[path] = path_stats

            return {
                'samples': samples,
                'stats': result
            }

        return wrapper

    return decorator


# ---- 应用装饰器的 DataSampler 函数 ----
@stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def DataSampler(n_samples=1, **kwargs):
    """生成结构化的随机测试数据样本集"""
    structure = kwargs.pop('structure', {})  # 从 kwargs 中移除 structure
    return [generate_nested_structure(structure, **kwargs) for _ in range(n_samples)]


# ---- 使用示例 ----
if __name__ == "__main__":
    # 定义数据结构
    user_structure = {
        'id': int,
        'name': str,
        'age': {'type': int, 'min': 18, 'max': 99},
        'is_active': bool,
        'created_at': datetime,
        'tags': [str],
        'details': {
            'address': str,
            'scores': (float, float, float)
        }
    }

    # 生成样本并计算统计数据
    result = DataSampler(n_samples=5, structure=user_structure)

    # 打印样本数据
    for i, sample in enumerate(result['samples']):
        print(f"样本 {i + 1}:")
        print(sample)
        print("-" * 50)

    # 打印统计结果
    print("\n统计结果:")
    for path, stats in result['stats'].items():
        print(f"路径: {path}")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {stat_value:.4f}")
        print("-" * 30)
