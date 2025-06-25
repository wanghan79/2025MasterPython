import random
import string
from datetime import date, timedelta

def generate_random_value(data_type, **kwargs):
    """根据指定的数据类型生成随机值"""
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
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    elif data_type == bool:
        return random.choice([True, False])
    elif data_type == date:
        start_date = kwargs.get('start_date', date(2000, 1, 1))
        end_date = kwargs.get('end_date', date.today())
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        return start_date + timedelta(days=random_days)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def DataSampler(**kwargs):
    """
    生成结构化的随机样本数据
    
    参数:
        **kwargs: 定义样本结构的关键字参数
            - n_samples: 生成的样本数量 (默认: 1)
            - structure: 样本的数据结构定义
    """
    n_samples = kwargs.get('n_samples', 1)
    structure = kwargs.get('structure', {})
    samples = []
    
    for _ in range(n_samples):
        sample = _build_sample(structure)
        samples.append(sample)
    
    return samples

def _build_sample(structure):
    """根据结构定义递归构建样本"""
    if isinstance(structure, dict):
        result = {}
        for key, value in structure.items():
            if isinstance(value, dict) and '__type__' in value:
                # 处理类型定义
                data_type = value['__type__']
                if data_type in (list, tuple):
                    element_type = value.get('__element_type__', str)
                    min_len = value.get('min_length', 0)
                    max_len = value.get('max_length', 10)
                    length = random.randint(min_len, max_len)
                    elements = [_build_sample({'__type__': element_type}) for _ in range(length)]
                    result[key] = tuple(elements) if data_type == tuple else elements
                elif data_type == dict:
                    dict_structure = value.get('__structure__', {})
                    result[key] = _build_sample(dict_structure)
                else:
                    # 基本数据类型
                    result[key] = generate_random_value(data_type, **value)
            else:
                # 嵌套结构
                result[key] = _build_sample(value)
        return result
    elif isinstance(structure, list) and len(structure) == 1:
        # 列表结构定义
        element_type = structure[0]
        length = random.randint(1, 5)  # 默认列表长度1-5
        return [_build_sample(element_type) for _ in range(length)]
    else:
        # 基本数据类型或未知结构
        return generate_random_value(structure) if isinstance(structure, type) else structure

# 示例用法
if __name__ == "__main__":
    # 示例1: 生成用户数据
    print('示例: 随机生成包含嵌套数据结构的用户数据')
    user_structure = {
        'id': {'__type__': int},
        'name': {'__type__': str, 'length': 8},
        'age': {'__type__': int, 'min': 18, 'max': 60},
        'is_active': {'__type__': bool},
        'registration_date': {'__type__': date},  # 使用 date 类型
        'address': {
            '__type__': dict,
            '__structure__': {
                'street': {'__type__': str},
                'city': {'__type__': str},
                'zip_code': {'__type__': str, 'length': 5}
            }
        },
        'hobbies': {
            '__type__': list,
            '__element_type__': str
        }
    }
    
    users = DataSampler(n_samples=3, structure=user_structure)
    for user in users:
        print(user)