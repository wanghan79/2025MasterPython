import random
import string
from datetime import datetime, timedelta

def DataSampler(**kwargs):
    dtype = kwargs.get('type')
    if dtype == 'int':
        return random.randint(*kwargs.get('range', (0, 100)))
    elif dtype == 'float':
        return random.uniform(*kwargs.get('range', (0, 100)))
    elif dtype == 'str':
        length = kwargs.get('length', 10)
        chars = kwargs.get('chars', string.ascii_letters + string.digits)
        return ''.join(random.choices(chars, k=length))
    elif dtype == 'bool':
        return random.choice([True, False])
    elif dtype == 'date':
        start = kwargs.get('start', datetime(2000, 1, 1))
        end = kwargs.get('end', datetime(2023, 1, 1))
        return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))
    elif dtype == 'list':
        return [DataSampler(**kwargs['item']) for _ in range(kwargs.get('size', 3))]
    elif dtype == 'tuple':
        return tuple(DataSampler(**kwargs['item']) for _ in range(kwargs.get('size', 3)))
    elif dtype == 'dict':
        return {k: DataSampler(**v) for k, v in kwargs['fields'].items()}
    return None

# 示例：生成包含嵌套结构的数据
user_data = {
    'type': 'dict',
    'fields': {
        'id': {'type': 'int', 'range': (1, 1000)},
        'name': {'type': 'str', 'length': 8},
        'scores': {'type': 'list', 'item': {'type': 'float', 'range': (0, 100)}, 'size': 3},
        'active': {'type': 'bool'},
        }
    }

print(DataSampler(**user_data))