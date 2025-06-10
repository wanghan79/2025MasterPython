import random
import string
import datetime

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_date(start_year=2000, end_year=2025):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    return start + datetime.timedelta(days=random.randint(0, (end - start).days))

def generate_value(dtype):
    if dtype == 'int':
        return random.randint(0, 100)
    elif dtype == 'float':
        return round(random.uniform(0, 100), 2)
    elif dtype == 'str':
        return random_string()
    elif dtype == 'bool':
        return random.choice([True, False])
    elif dtype == 'date':
        return random_date()
    elif isinstance(dtype, dict):
        return generate_structure(dtype)
    elif isinstance(dtype, list):
        return [generate_value(t) for t in dtype]
    elif isinstance(dtype, tuple):
        return tuple(generate_value(t) for t in dtype)
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

def generate_structure(struct_def):
    if isinstance(struct_def, dict):
        return {k: generate_value(v) for k, v in struct_def.items()}
    elif isinstance(struct_def, list):
        return [generate_value(t) for t in struct_def]
    elif isinstance(struct_def, tuple):
        return tuple(generate_value(t) for t in struct_def)
    else:
        return generate_value(struct_def)

def generate_samples(num=1, **kwargs):
    """
    :param num: 样本数量
    :param kwargs: 指定结构，例如 user={'id': 'int', 'name': 'str', 'info': {'age': 'int', 'birth': 'date'}}
    :return: 随机样本列表
    """
    structure = kwargs.get("structure", {})
    return [generate_structure(structure) for _ in range(num)]


samples = generate_samples(
    num=2,
    structure={
        'user': {
            'id': 'int',
            'profile': {
                'name': 'str',
                'birthday': 'date',
                'hobbies': ['str', 'str']
            },
            'scores': ('float', 'float', 'float'),
        },
        'is_member': 'bool'
    }
)

for s in samples:
    print(s)

