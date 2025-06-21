import random
import string
import datetime

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_date(start_year=2000, end_year=2025):
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    delta_days = (end_date - start_date).days
    return start_date + datetime.timedelta(days=random.randint(0, delta_days))

def generate_value(value_spec):
    """
    根据value_spec生成一个随机值，可以是嵌套结构
    """
    if isinstance(value_spec, str):
        if value_spec == 'int':
            return random.randint(0, 100)
        elif value_spec == 'float':
            return round(random.uniform(0, 100), 2)
        elif value_spec == 'str':
            return random_string()
        elif value_spec == 'bool':
            return random.choice([True, False])
        elif value_spec == 'date':
            return random_date()
        else:
            raise ValueError(f"Unsupported basic type: {value_spec}")
    
    elif isinstance(value_spec, list):
        # 结构: ['list', element_spec, count]
        if value_spec[0] == 'list':
            return [generate_value(value_spec[1]) for _ in range(value_spec[2])]
        # 结构: ['tuple', element1_spec, element2_spec, ...]
        elif value_spec[0] == 'tuple':
            return tuple(generate_value(sub) for sub in value_spec[1:])
        else:
            raise ValueError(f"Unsupported list structure: {value_spec}")
    
    elif isinstance(value_spec, dict):
        # 嵌套字典结构
        return {k: generate_value(v) for k, v in value_spec.items()}
    
    else:
        raise TypeError(f"Unsupported spec type: {type(value_spec)}")


def data_sampler(sample_count=1, **structure_spec):
    """
    生成结构化的随机数据样本
    :param sample_count: 样本数量
    :param structure_spec: 每个字段的数据类型或嵌套结构说明
    :return: List of structured samples
    """
    samples = []
    for _ in range(sample_count):
        sample = {key: generate_value(spec) for key, spec in structure_spec.items()}
        samples.append(sample)
    return samples

if __name__ == "__main__":
    # 定义嵌套结构
    schema = {
        "id": "int",
        "name": "str",
        "signup": "date",
        "active": "bool",
        "profile": {
            "age": "int",
            "score": "float",
            "tags": ['list', 'str', 3],
            "location": {
                "lat": "float",
                "lng": "float"
            }
        },
        "history": ['list', {'time': 'date', 'action': 'str'}, 2],
        "device": ['tuple', "str", "bool", "float"]
    }

    # 生成 3 条样本数据
    result = data_sampler(3, **schema)
    for item in result:
        print(item)
