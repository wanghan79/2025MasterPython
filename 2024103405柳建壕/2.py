import random
import string
from datetime import datetime, timedelta


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


def DataSampler(structure, num_samples=1, **kwargs):
    """
    生成结构化的随机测试数据样本集

    - structure: 数据结构模板，可以是字典、列表、元组或基本数据类型
    - num_samples: 要生成的样本数量
    - **kwargs: 其他可选参数，如数值范围、字符串长度等

    返回生成的样本集列表
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

    # 生成5个用户样本
    users = DataSampler(
        user_structure,
        num_samples=5,
        min=0,  # 数值最小值
        max=100,  # 数值最大值
        length=8,  # 字符串长度
        min_length=2  # 列表最小长度
    )

    # 打印生成的样本
    for i, user in enumerate(users, 1):
        print(f"用户样本 {i}:")
        for key, value in user.items():
            print(f"  {key}: {value}")
        print()
