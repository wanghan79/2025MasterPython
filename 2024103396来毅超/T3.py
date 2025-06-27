import random
import string
import math
from functools import wraps

class User:
    def __init__(self, uid, name, active):
        self.uid = uid
        self.name = name
        self.active = active

    def __repr__(self):
        return f"User(uid={self.uid}, name='{self.name}', active={self.active})"

class Product:
    def __init__(self, pid, pname, price):
        self.pid = pid
        self.pname = pname
        self.price = price

    def __repr__(self):
        return f"Product(pid={self.pid}, pname='{self.pname}', price={self.price})"

def extract_numbers(data):
    result = []
    if isinstance(data, bool):
        return result
    elif isinstance(data, (int, float)):
        return [data]
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            result.extend(extract_numbers(item))
    elif isinstance(data, dict):
        for k, v in data.items():
            result.extend(extract_numbers(k))
            result.extend(extract_numbers(v))
    elif hasattr(data, '__dict__'):
        for attr in vars(data).values():
            result.extend(extract_numbers(attr))
    return result

def calculate_statistics(values, metrics):
    stats = {}
    if not values:
        return {metric: None for metric in metrics}
    avg = sum(values) / len(values)
    if 'sum' in metrics:
        stats['sum'] = sum(values)
    if 'mean' in metrics:
        stats['mean'] = avg
    if 'var' in metrics:
        stats['var'] = sum((x - avg) ** 2 for x in values) / len(values)
    if 'rmse' in metrics:
        stats['rmse'] = math.sqrt(sum(x ** 2 for x in values) / len(values))
    return stats

def with_statistics(*metrics):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sample = func(*args, **kwargs)
            values = extract_numbers(sample)
            return calculate_statistics(values, metrics)
        return wrapper
    return decorator

@with_statistics('sum', 'mean', 'var', 'rmse')
def generate_sample(**schema):
    return data_sampling(**schema)

def data_sampling(**schema):
    def gen_int(cfg): return random.randint(*cfg['datarange'])
    def gen_float(cfg): return random.uniform(*cfg['datarange'])
    def gen_str(cfg): return ''.join(random.choices(cfg['datarange'], k=cfg['len']))
    def gen_bool(_): return random.choice([True, False])

    def gen_list(cfg):
        count, elem = cfg['len'], cfg['element']
        if isinstance(elem, list):
            return [data_sampling(**e) for e in elem]
        return [data_sampling(**elem) for _ in range(count)]

    def gen_tuple(cfg):
        count, elem = cfg['len'], cfg['element']
        if isinstance(elem, list):
            return tuple(data_sampling(**e) for e in elem)
        return tuple(data_sampling(**elem) for _ in range(count))

    def gen_dict(cfg):
        out = {}
        entries, count = cfg['entries'], cfg['len']
        if isinstance(entries, list):
            for item in entries:
                k = handler[next(iter(item['key']))](item['key'][next(iter(item['key']))])
                v = handler[next(iter(item['value']))](item['value'][next(iter(item['value']))])
                out[k] = v
        else:
            k_cfg = entries['key']
            v_cfg = entries['value']
            keys_seen = set()
            while len(out) < count:
                key = handler[next(iter(k_cfg))](k_cfg[next(iter(k_cfg))])
                if key in keys_seen:
                    continue
                val = handler[next(iter(v_cfg))](v_cfg[next(iter(v_cfg))])
                out[key] = val
                keys_seen.add(key)
        return out

    def gen_object(cfg):
        cls, attrs = cfg['class'], cfg['attrs']
        args = {name: handler[next(iter(tdesc))](tdesc[next(iter(tdesc))]) for name, tdesc in attrs.items()}
        return cls(**args)

    handler = {
        'int': gen_int, 'float': gen_float, 'str': gen_str, 'bool': gen_bool,
        'list': gen_list, 'tuple': gen_tuple, 'dict': gen_dict, 'object': gen_object
    }

    return [handler[typ](cfg) for typ, cfg in schema.items()][0 if len(schema) == 1 else slice(None)]

# Sample structure
schema = {
    "int": {"datarange": (1, 100)},
    "list": {
        "len": 3,
        "element": {"float": {"datarange": (0.0, 10.0)}}
    }
}

result = generate_sample(**schema)
print(result)
"""
{'sum': 69.67437120602315, 'mean': 17.41859280150579, 'var': 405.1232580472275, 'rmse': 26.618238732716648}
"""
