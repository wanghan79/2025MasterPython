import random
import string

def dataSampling(**kwargs):
    """
    dataSampling(**kwargs)

    This function recursively generates randomized data based on a user-defined structural schema.
    It supports basic types (int, float, str, bool), compound types (list, tuple, dict),
    and user-defined Python objects.

    Arguments:
        kwargs: A schema describing the structure of data to generate.

    Returns:
        A single sample


    Supported type descriptions:

    # Integer type
    "int": {
        "datarange": (min: int, max: int)  # inclusive range
    },

    # Float type
    "float": {
        "datarange": (min: float, max: float)  # inclusive range
    },

    # String type
    "str": {
        "datarange": "charset",   # allowed characters
        "len": length: int        # string length
    },

    # Boolean type
    "bool": {}

    # List type
    "list": {
        "len": N: int,  # number of elements to generate

        # element can be:
        # - a single type description → repeated N times (homogeneous list)
        # - a list of type descriptions → must match length N (heterogeneous list)
        # each element can itself be any type, including nested list/tuple/dict/object
        "element": { <type> } or [ <type1>, <type2>, ... ]
    },

    # Tuple type
    "tuple": {
        "len": N: int,  # number of elements in the tuple

        # element format same as list
        # each tuple element can be any supported type, including nested combinations
        "element": { <type> } or [ <type1>, <type2>, ... ]
    },

    # Dictionary type
    "dict": {
        "len": N: int,  # total number of key-value pairs to generate

        # entries can be:
        # - a single {key, value} structure → repeat N times (homogeneous dict)
        # - a list of {key, value} structures → must match len=N (heterogeneous dict)
        #
        # key: any supported basic type (str, int, tuple, etc.)
        # value: any supported type, including nested list / tuple / dict / object
        "entries": {
            "key":   { <key_type> },
            "value": { <value_type> }
        }
        # or
        "entries": [
            {
                "key":   { <key_type> },
                "value": { <value_type> }
            },
            ...
        ]
    }

    # Object type (custom class)
    "object": {
        "class": ClassReference,  # Python class to instantiate

        # each attribute is a field name mapped to a type description
        # values can be nested types including list/tuple/dict/object
        "attrs": {
            "field1": { <type> },
            "field2": { <type> },
            ...
        }
    }
    """

    # Handle integer generation with specified range
    def handle_int(desc):
        # Get the range for integers
        datarange = desc.get('datarange')
        # Validate the range is a tuple of two values
        if not isinstance(datarange, tuple) or len(datarange) != 2:
            raise ValueError("int type requires 'datarange' as a tuple (min, max)")
        # Return a random integer in the range [min, max]
        return random.randint(*datarange)

    # Handle float generation with specified range
    def handle_float(desc):
        # Get the range for floats
        datarange = desc.get('datarange')
        # Validate format
        if not isinstance(datarange, tuple) or len(datarange) != 2:
            raise ValueError("float type requires 'datarange' as a tuple (min, max)")
        # Return a random float in the range [min, max)
        return random.uniform(*datarange)

    # Handle string generation with character set and length
    def handle_str(desc):
        # Get allowed characters and target length
        charset = desc.get('datarange')
        length = desc.get('len')
        # Validate input
        if not isinstance(charset, str):
            raise ValueError("str type requires 'datarange' to be a string")
        if not isinstance(length, int) or length <= 0:
            raise ValueError("str type requires positive 'len'")
        # Randomly generate a string of specified length from given charset
        return ''.join(random.choices(charset, k=length))

    # Handle random boolean generation
    def handle_bool(desc):
        # Return a random boolean value (True or False)
        return random.choice([True, False])

    # Handle list generation: homogeneous or heterogeneous element structure
    def handle_list(desc):
        length = desc.get('len')
        element_desc = desc.get('element')
        # Validate length and element descriptor
        if not isinstance(length, int) or length <= 0:
            raise ValueError("list must specify positive 'len'")
        if element_desc is None:
            raise ValueError("list must specify 'element'")
        # Case 1: Heterogeneous list - element is a list of type descriptions
        if isinstance(element_desc, list):
            if len(element_desc) != length:
                raise ValueError("Length of list must match number of element descriptions")
            return [dataSampling(**e) for e in element_desc]
        # Case 2: Homogeneous list - repeat single structure
        elif isinstance(element_desc, dict):
            return [dataSampling(**element_desc) for _ in range(length)]
        else:
            raise ValueError("list 'element' must be a dict or list")

    # Handle tuple generation: homogeneous or heterogeneous element structure
    def handle_tuple(desc):
        length = desc.get('len')
        element_desc = desc.get('element')
        # Validate inputs
        if not isinstance(length, int) or length <= 0:
            raise ValueError("tuple must specify positive 'len'")
        if element_desc is None:
            raise ValueError("tuple must specify 'element'")
        # Case 1: Heterogeneous tuple
        if isinstance(element_desc, list):
            if len(element_desc) != length:
                raise ValueError("Length of tuple must match number of element descriptions")
            return tuple(dataSampling(**e) for e in element_desc)
        # Case 2: Homogeneous tuple
        elif isinstance(element_desc, dict):
            return tuple(dataSampling(**element_desc) for _ in range(length))
        else:
            raise ValueError("tuple 'element' must be a dict or list")

    # Handle dictionary generation: support fixed entry list or repeated structure
    def handle_dict(desc):
        entries = desc.get('entries')
        total = desc.get('len')
        # Validate inputs
        if entries is None or total is None:
            raise ValueError("dict must specify both 'len' and 'entries'")
        if not isinstance(total, int) or total <= 0:
            raise ValueError("'len' in dict must be a positive integer")
        result = {}
        # Case 1: entries is a list → fixed structure (heterogeneous dict)
        if isinstance(entries, list):
            if len(entries) != total:
                raise ValueError(f"dict.entries length ({len(entries)}) must match 'len' ({total})")
            for entry in entries:
                key_desc = entry.get("key")
                value_desc = entry.get("value")
                # Each key/value must be a single-type dict
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

        # Case 2: entries is a dict → repeat N times, ensure unique keys
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
            attempt_limit = total * 5 # prevent infinite loop if key collisions occur

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

    # Handle instantiation of user-defined Python objects
    def handle_object(desc):
        cls = desc.get('class')
        attrs = desc.get('attrs')
        # Validate class and attribute definitions
        if cls is None or not callable(cls):
            raise ValueError("object must specify a valid 'class'")
        if not isinstance(attrs, dict):
            raise ValueError("object must specify 'attrs' as a dict")
        attr_values = {}
        for attr_name, attr_desc in attrs.items():
            # Each attribute must be defined as a single-type structure
            if not isinstance(attr_desc, dict) or len(attr_desc) != 1:
                raise ValueError(f"Invalid attr description for '{attr_name}'")
            type_name, type_desc = next(iter(attr_desc.items()))
            if type_name not in handler_map:
                raise ValueError(f"Unsupported type in object attribute: {type_name}")
            attr_values[attr_name] = handler_map[type_name](type_desc)
        # Instantiate object with generated attributes
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

    """
    Generate a single sample based on the provided structure.

    Returns:
        - A single value (if the structure contains only one type)
        - A list of values (if the structure contains multiple types)
    """
    results = []
    for type_name, desc in kwargs.items():
        # Ensure the type is supported
        if type_name not in handler_map:
            raise ValueError(f"Unsupported type: {type_name}")
        # Call the corresponding handler to generate data for this type
        results.append(handler_map[type_name](desc))
    # Return a single value if only one top-level type is defined
    return results[0] if len(results) == 1 else results

#  ===== example custom class =====
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

sample = dataSampling(**struct2)
print(sample)