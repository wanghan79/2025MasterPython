import random
import string
from datetime import datetime, timedelta


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


def DataSampler(n_samples=1, **kwargs):
    structure = kwargs.pop('structure', {})  # 从 kwargs 中移除 structure
    return [generate_nested_structure(structure, **kwargs) for _ in range(n_samples)]


# 使用示例
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

    # 生成5个用户样本
    samples = DataSampler(n_samples=5, structure=user_structure)

    # 打印结果
    for i, sample in enumerate(samples):
        print(f"样本 {i + 1}:")
        print(sample)
        print("-" * 50)