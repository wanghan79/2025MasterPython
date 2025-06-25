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
    """
       Recursively extract all numeric values (int, float) from the input data structure.
       Ignores boolean values since bool is a subclass of int in Python.
       Supports nested containers (list, tuple, set, dict, object).
    """
    nums = []
    if isinstance(data, bool):
        return nums  # Exclude bool values
    if isinstance(data, (int, float)):
        nums.append(data)
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            nums.extend(extract_numbers(item)) # Recurse into iterable
    elif isinstance(data, dict):
        for k, v in data.items():
            nums.extend(extract_numbers(k)) # Recurse into keys
            nums.extend(extract_numbers(v)) # Recurse into values
    elif hasattr(data, '__dict__'):
        # Recurse into attributes of user-defined objects
        for attr in vars(data).values():
            nums.extend(extract_numbers(attr))
    return nums

def calculate_statistics(numbers, stat_types):
    """
        Calculate the specified statistics over a list of numbers.
        Supported statistics: 'sum', 'mean', 'var', 'rmse'.
        Returns a dictionary with selected metrics.
    """
    if not numbers:
        return {stat: None for stat in stat_types}  # Handle empty case
    result = {}
    if 'sum' in stat_types:
        result['sum'] = sum(numbers)
    if 'mean' in stat_types:
        result['mean'] = sum(numbers) / len(numbers)
    if 'var' in stat_types:
        mean = sum(numbers) / len(numbers)
        result['var'] = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    if 'rmse' in stat_types:
        result['rmse'] = math.sqrt(sum(x ** 2 for x in numbers) / len(numbers))
    return result

def statisticsRes(*stat_types):
    """
       A decorator factory that adds numeric statistics computation
       to any function that returns structured data.
       The statistics types (e.g., 'sum', 'mean') are passed as arguments.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs) # Call the wrapped function
            print("data",data)
            numbers = extract_numbers(data) # Extract numeric fields
            print("numbers", numbers)
            stats = calculate_statistics(numbers, stat_types)
            print("stats")
            return stats  # Return final statistics dict
        return wrapper
    return decorator

@statisticsRes('sum', 'mean', 'var', 'rmse')
def printSamples(**kwargs):
    """
       Calls dataSampling with the provided structure and applies statistical analysis
       using the statisticsRes decorator.
    """
    return dataSampling(**kwargs)

def dataSampling(**kwargs):
    def handle_int(desc):
        datarange = desc.get('datarange')
        if not isinstance(datarange, tuple) or len(datarange) != 2:
            raise ValueError("int type requires 'datarange' as a tuple (min, max)")
        return random.randint(*datarange)

    def handle_float(desc):
        datarange = desc.get('datarange')
        if not isinstance(datarange, tuple) or len(datarange) != 2:
            raise ValueError("float type requires 'datarange' as a tuple (min, max)")
        return random.uniform(*datarange)

    def handle_str(desc):
        charset = desc.get('datarange')
        length = desc.get('len')
        if not isinstance(charset, str):
            raise ValueError("str type requires 'datarange' to be a string")
        if not isinstance(length, int) or length <= 0:
            raise ValueError("str type requires positive 'len'")
        return ''.join(random.choices(charset, k=length))

    def handle_bool(desc):
        return random.choice([True, False])

    def handle_list(desc):
        length = desc.get('len')
        element_desc = desc.get('element')
        if not isinstance(length, int) or length <= 0:
            raise ValueError("list must specify positive 'len'")
        if element_desc is None:
            raise ValueError("list must specify 'element'")
        if isinstance(element_desc, list):
            if len(element_desc) != length:
                raise ValueError("Length of list must match number of element descriptions")
            return [dataSampling(**e) for e in element_desc]
        elif isinstance(element_desc, dict):
            return [dataSampling(**element_desc) for _ in range(length)]
        else:
            raise ValueError("list 'element' must be a dict or list")

    def handle_tuple(desc):
        length = desc.get('len')
        element_desc = desc.get('element')
        if not isinstance(length, int) or length <= 0:
            raise ValueError("tuple must specify positive 'len'")
        if element_desc is None:
            raise ValueError("tuple must specify 'element'")
        if isinstance(element_desc, list):
            if len(element_desc) != length:
                raise ValueError("Length of tuple must match number of element descriptions")
            return tuple(dataSampling(**e) for e in element_desc)
        elif isinstance(element_desc, dict):
            return tuple(dataSampling(**element_desc) for _ in range(length))
        else:
            raise ValueError("tuple 'element' must be a dict or list")

    def handle_dict(desc):
        entries = desc.get('entries')
        total = desc.get('len')
        if entries is None or total is None:
            raise ValueError("dict must specify both 'len' and 'entries'")
        if not isinstance(total, int) or total <= 0:
            raise ValueError("'len' in dict must be a positive integer")
        result = {}
        if isinstance(entries, list):
            if len(entries) != total:
                raise ValueError(f"dict.entries length ({len(entries)}) must match 'len' ({total})")
            for entry in entries:
                key_desc = entry.get("key")
                value_desc = entry.get("value")
                if not isinstance(key_desc, dict) or len(key_desc) != 1:
                    raise ValueError(f"Invalid key description: {key_desc}")
                if not isinstance(value_desc, dict) or len(value_desc) != 1:
                    raise ValueError(f"Invalid value description: {value_desc}")
                key_type, key_type_desc = next(iter(key_desc.items()))
                value_type, value_type_desc = next(iter(value_desc.items()))
                if key_type not in handler_map or value_type not in handler_map:
                    raise ValueError(f"Unsupported key/value type: {key_type}, {value_type}")
                key = handler_map[key_type](key_type_desc)
                value = handler_map[value_type](value_type_desc)
                result[key] = value

        elif isinstance(entries, dict):
            key_desc = entries.get("key")
            value_desc = entries.get("value")
            if not isinstance(key_desc, dict) or len(key_desc) != 1:
                raise ValueError(f"Invalid key description: {key_desc}")
            if not isinstance(value_desc, dict) or len(value_desc) != 1:
                raise ValueError(f"Invalid value description: {value_desc}")
            key_type, key_type_desc = next(iter(key_desc.items()))
            value_type, value_type_desc = next(iter(value_desc.items()))
            if key_type not in handler_map or value_type not in handler_map:
                raise ValueError(f"Unsupported key/value type: {key_type}, {value_type}")

            generated_keys = set()
            attempts = 0
            attempt_limit = total * 5

            while len(result) < total and attempts < attempt_limit:
                key = handler_map[key_type](key_type_desc)
                if key in generated_keys:
                    attempts += 1
                    continue
                value = handler_map[value_type](value_type_desc)
                result[key] = value
                generated_keys.add(key)

            if len(result) < total:
                raise ValueError("Unable to generate enough unique keys for dict")
        else:
            raise ValueError("dict 'entries' must be a dict or list")

        return result

    def handle_object(desc):
        cls = desc.get('class')
        attrs = desc.get('attrs')
        if cls is None or not callable(cls):
            raise ValueError("object must specify a valid 'class'")
        if not isinstance(attrs, dict):
            raise ValueError("object must specify 'attrs' as a dict")
        attr_values = {}
        for attr_name, attr_desc in attrs.items():
            if not isinstance(attr_desc, dict) or len(attr_desc) != 1:
                raise ValueError(f"Invalid attr description for '{attr_name}'")
            type_name, type_desc = next(iter(attr_desc.items()))
            if type_name not in handler_map:
                raise ValueError(f"Unsupported type in object attribute: {type_name}")
            attr_values[attr_name] = handler_map[type_name](type_desc)
        return cls(**attr_values)

    handler_map = {
        'int': handle_int,
        'float': handle_float,
        'str': handle_str,
        'bool': handle_bool,
        'list': handle_list,
        'tuple': handle_tuple,
        'dict': handle_dict,
        'object': handle_object
    }
    results = []
    for type_name, desc in kwargs.items():
        if type_name not in handler_map:
            raise ValueError(f"Unsupported type: {type_name}")
        results.append(handler_map[type_name](desc))
    return results[0] if len(results) == 1 else results

# ===== example structure =====
struct = {"int":{"datarange":(0,100)},"float":{"datarange":(0,100)},"str":{"datarange":string.ascii_uppercase,'len':50} }

struct1 = {
    "dict": {
        "len": 3,
        "entries": [
            {
                "key": {"str": {"datarange": string.ascii_lowercase, "len": 5}},
                "value": {"int": {"datarange": (0, 100)}}
            },
            {
                "key": {"int": {"datarange": (1000, 9999)}},
                "value": {"object": {
                    "class": User,
                    "attrs": {
                        "uid": {"int": {"datarange": (1, 9999)}},
                        "name": {"str": {"datarange": string.ascii_letters, "len": 6}},
                        "active": {"bool": {}}
                    }
                }}
            },
            {
                "key": {"str": {"datarange": "abcde", "len": 3}},
                "value": {"list": {
                    "len": 3,
                    "element": {"float": {"datarange": (0.0, 1.0)}}
                }}
            }
        ]
    }
}

struct2 = {
    "int": {"datarange": (1, 100)},
    "list": {
        "len": 8,
        "element": [
            {"int": {"datarange": (1, 100)}},
            {"float": {"datarange": (0.0, 100.0)}},
            {"str": {"datarange": string.ascii_letters, "len": 6}},
            {"bool": {}},
            {"tuple": {
                "len": 3,
                "element": {"bool": {}}
            }},
            {"list": {
                "len": 3,
                "element": {"int": {"datarange": (10, 99)}}
            }},
            {"dict": {
                "len": 2,
                "entries":
                    {
                        "key": {"str": {"datarange": "xyz", "len": 3}},
                        "value": {"float": {"datarange": (0.0, 10.0)}}
                    }
            }},
            {"object": {
                "class": User,
                "attrs": {
                    "uid": {"int": {"datarange": (1000, 9999)}},
                    "name": {"str": {"datarange": string.ascii_letters, "len": 6}},
                    "active": {"bool": {}}
                }
            }}
        ]
    }
}

struct3 = {
    "int": {"datarange": (1, 100)},
    "float": {"datarange": (0.0, 100.0)},
    "str": {"datarange": string.ascii_letters, "len": 6},
    "bool":{},
    "tuple": {
        "len": 3,
        "element": [
            {"int": {"datarange": (1, 10)}},
            {"list": {
                "len": 2,
                "element": {"str": {"datarange": string.ascii_lowercase, "len": 4}},
            }},
            {"dict": {
                "len": 2,
                "entries": [
                    {
                        "key": {"str": {"datarange": string.ascii_lowercase, "len": 3}},
                        "value": {"bool": {}}
                    },
                    {
                        "key": {"int": {"datarange": (100, 200)}},
                        "value": {"float": {"datarange": (0.0, 10.0)}}
                    }
                ]
            }}
        ]
    },
    "dict": {
        "len": 3,
        "entries": [
            {
                "key": {"str": {"datarange": "abc", "len": 3}},
                "value": {"tuple": {
                    "len": 2,
                    "element": [
                        {"int": {"datarange": (10, 20)}},
                        {"float": {"datarange": (0.1, 0.9)}}
                    ]
                }}
            },
            {
                "key": {"int": {"datarange": (1000, 2000)}},
                "value": {"object": {
                    "class": User,
                    "attrs": {
                        "uid": {"int": {"datarange": (1, 9999)}},
                        "name": {"str": {"datarange": string.ascii_letters, "len": 5}},
                        "active": {"bool": {}}
                    }
                }}
            },
            {
                "key": {"tuple": {
                    "len": 2,
                    "element": [
                        {"str": {"datarange": "xyz", "len": 2}},
                        {"int": {"datarange": (1, 10)}}
                    ]
                }},
                "value": {"list": {
                    "len": 2,
                    "element": {"object": {
                        "class": Product,
                        "attrs": {
                            "pid": {"int": {"datarange": (100, 999)}},
                            "pname": {"str": {"datarange": string.ascii_uppercase, "len": 4}},
                            "price": {"float": {"datarange": (1.0, 100.0)}}
                        }
                    }}
                }}
            }
        ]
    }
}

result = printSamples(**struct2)
print(result)