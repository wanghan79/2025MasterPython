import random
import string
import datetime
from typing import List, Dict, Union, Tuple, Any


def random_string(length: int) -> str:
    """生成随机字符串"""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))


def random_date(start_date: datetime.date, end_date: datetime.date) -> datetime.date:
    """在给定的日期范围内生成随机日期"""
    delta = end_date - start_date
    return start_date + datetime.timedelta(days=random.randint(0, delta.days))


def random_list(length: int, data_type: str, **kwargs) -> List[Any]:
    """生成随机列表"""
    return [generate_random_data(data_type, **kwargs) for _ in range(length)]


def random_tuple(length: int, data_type: str, **kwargs) -> Tuple[Any, ...]:
    """生成随机元组"""
    return tuple(random_list(length, data_type, **kwargs))


def random_dict(keys: List[str], values: List[Any]) -> Dict[str, Any]:
    """生成随机字典"""
    return {k: v for k, v in zip(keys, values)}


def generate_random_data(data_type: str, **kwargs) -> Any:
    """根据数据类型生成随机数据"""
    if data_type == 'int':
        return random.randint(kwargs.get('min', 0), kwargs.get('max', 100))
    elif data_type == 'float':
        return random.uniform(kwargs.get('min', 0.0), kwargs.get('max', 100.0))
    elif data_type == 'str':
        return random_string(kwargs.get('length', 10))
    elif data_type == 'bool':
        return random.choice([True, False])
    elif data_type == 'date':
        start_date = kwargs.get('start_date', datetime.date(2000, 1, 1))
        end_date = kwargs.get('end_date', datetime.date.today())
        return random_date(start_date, end_date)
    elif data_type == 'list':
        data_type = kwargs.get('data_type', 'int')
        length = kwargs.get('length', 5)
        # 移除 length 键，避免重复传递
        kwargs.pop('length', None)
        return random_list(length, data_type, **kwargs)
    elif data_type == 'tuple':
        data_type = kwargs.get('data_type', 'int')
        length = kwargs.get('length', 5)
        # 移除 length 键，避免重复传递
        kwargs.pop('length', None)
        return random_tuple(length, data_type, **kwargs)
    elif data_type == 'dict':
        keys = kwargs.get('keys', ['key1', 'key2', 'key3'])
        value_types = kwargs.get('value_types', ['int', 'float', 'str'])
        values = [generate_random_data(t, **kwargs) for t in value_types]
        return random_dict(keys, values)
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")


def data_sampler(sample_size: int, **kwargs) -> List[Dict[str, Any]]:
    """生成结构化的模拟数据"""
    samples = []
    for _ in range(sample_size):
        sample = {}
        for key, value in kwargs.items():
            data_type = value.get('type')
            # 从 value 字典中移除 'type' 键，避免重复传递给 generate_random_data
            params = {k: v for k, v in value.items() if k != 'type'}
            # 确保 params 中不包含 data_type 键
            params.pop('data_type', None)
            sample[key] = generate_random_data(data_type, **params)
        samples.append(sample)
    return samples


# 示例用法
if __name__ == "__main__":
    sample_size = 5  # 生成5个样本
    user_data = data_sampler(
        sample_size,
        name={'type': 'str', 'length': 10},
        age={'type': 'int', 'min': 18, 'max': 100},
        height={'type': 'float', 'min': 150.0, 'max': 200.0},
        is_active={'type': 'bool'},
        registered_date={'type': 'date', 'start_date': datetime.date(2000, 1, 1)},
        hobbies={'type': 'list', 'data_type': 'str', 'length': 3},
        scores={'type': 'dict', 'keys': ['math', 'english', 'science'], 'value_types': ['int', 'int', 'int']}
    )

    for idx, user in enumerate(user_data, 1):
        print(f"用户 {idx}:")
        for key, value in user.items():
            print(f"  {key}: {value}")
        print()