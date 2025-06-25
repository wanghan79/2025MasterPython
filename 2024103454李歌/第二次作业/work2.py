import random
import string
import datetime

def random_int():
    return random.randint(0, 100)

def random_float():
    return round(random.uniform(0, 100), 2)

def random_str(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_bool():
    return random.choice([True, False])

def random_date(start_year=2000, end_year=2025):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = (end - start).days
    return (start + datetime.timedelta(days=random.randint(0, delta))).isoformat()


# 类型映射
TYPE_GENERATORS = {
    'int': random_int,
    'float': random_float,
    'str': random_str,
    'bool': random_bool,
    'date': random_date
}

def generate_sample(structure):
    """
    根据结构递归生成一个随机样本
    """
    if isinstance(structure, dict):
        return {k: generate_sample(v) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [generate_sample(structure[0]) for _ in range(random.randint(1, 3))]
    elif isinstance(structure, tuple):
        return tuple(generate_sample(item) for item in structure)
    elif isinstance(structure, str):
        # 类型字段
        if structure.startswith("str:"):
            length = int(structure.split(":")[1])
            return random_str(length)
        elif structure in TYPE_GENERATORS:
            return TYPE_GENERATORS[structure]()
        else:
            raise ValueError(f"Unsupported type: {structure}")
    else:
        raise TypeError(f"Invalid structure type: {type(structure)}")

def DataSampler(sample_count=1, **kwargs):
    """
    生成指定数量的结构化样本数据
    参数：
        sample_count: 样本数量
        kwargs: 定义结构的嵌套结构体
    返回：
        样本列表
    """
    return [generate_sample(kwargs) for _ in range(sample_count)]
