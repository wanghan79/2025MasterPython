import random
import string

def random_value(type_spec):
    if type_spec == 'int':
        return random.randint(0, 100)
    elif type_spec == 'float':
        return round(random.uniform(0, 100), 2)
    elif type_spec == 'str':
        return ''.join(random.choices(string.ascii_letters, k=5))
    elif type_spec == 'bool':
        return random.choice([True, False])
    elif isinstance(type_spec, dict):
        return {k: random_value(v) for k, v in type_spec.items()}
    elif isinstance(type_spec, list):
        # [<type>, <length>] 表示 list 中元素的类型和数量
        value_type, length = type_spec
        return [random_value(value_type) for _ in range(length)]
    elif isinstance(type_spec, tuple):
        # (<type1>, <type2>, ...) 表示生成一个固定结构的 tuple
        return tuple(random_value(t) for t in type_spec)
    else:
        raise ValueError(f"Unsupported type spec: {type_spec}")

def generate_samples(sample_type: dict, **kwargs):
    sample_count = kwargs.get('sample_count', 1)
    return [random_value(sample_type) for _ in range(sample_count)]

# 定义嵌套数据结构样本格式
sample_schema = {
    'user_id': 'int',
    'username': 'str',
    'scores': ['float', 3],
    'profile': {
        'active': 'bool',
        'tags': ['str', 2],
        'flags': ('bool', 'bool')
    },
    'history': [
        {
            'timestamp': 'int',
            'action': 'str'
        },
        2
    ]
}


sample_data = generate_samples(sample_schema, sample_count=3)

for i, sample in enumerate(sample_data):
    print(f"样本 {i+1}:")
    print(sample)
    print('-' * 40)
