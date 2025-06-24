import math as m
import random as rnd
from pprint import pformat as pf
from functools import wraps as wr


def collect_statistics(*metrics):
    def decorator(func):
        @wr(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)

            calculations = {
                'total': lambda x: sum(x),
                'average': lambda x: sum(x) / len(x),
                'variation': lambda x: sum((i - sum(x) / len(x)) ** 2 for i in x) / len(x),
                'std_dev': lambda x: m.sqrt(sum((i - sum(x) / len(x)) ** 2 for i in x) / len(x)),
            }

            selected_calcs = {k: v for k, v in calculations.items()
                              if not metrics or k in metrics}

            stats_result = {}
            for record in data:
                for district in record.values():
                    for path, value in _unpack_dict(district):
                        if isinstance(value, (int, float)):
                            stats_result.setdefault(path, []).append(value)

            return {
                'raw_data': data,
                'analysis': {
                    path: {metric: calc(values)
                           for metric, calc in selected_calcs.items()}
                    for path, values in stats_result.items()
                }
            }

        return wrapper

    return decorator


def _unpack_dict(nested_dict, current_path=''):
    for key, val in nested_dict.items():
        new_path = f"{current_path}.{key}" if current_path else key
        if isinstance(val, dict):
            yield from _unpack_dict(val, new_path)
        else:
            yield new_path, val


def create_dataset(size, **template):
    dataset = []
    for i in range(size):
        entry = {}
        for name, structure in template.items():
            entry[f"{name}{i}"] = _construct(structure)
        dataset.append(entry)
    return dataset


def _construct(blueprint):
    if isinstance(blueprint, dict):
        return {k: _construct(v) for k, v in blueprint.items()}
    if isinstance(blueprint, (list, tuple)) and len(blueprint) == 2:
        low, high = blueprint
        if all(isinstance(x, int) for x in blueprint):
            return rnd.randint(low, high)
        if all(isinstance(x, float) for x in blueprint):
            return rnd.uniform(low, high)
    return blueprint


@collect_statistics('total', 'average', 'variation', 'std_dev')
def generate_city(num, **layout):
    return create_dataset(num, **layout)


if __name__ == "__main__":
    city_layout = {
        'district': {
            'education': {
                'staff': (50, 70),
                'pupils': (800, 1200),
                'support': (20, 40),
                'budget': (410000.5, 986553.1)
            },
            'healthcare': {
                'physicians': (40, 60),
                'staff': (60, 80),
                'clients': (200, 300),
                'funding': (110050.5, 426553.4)
            },
            'commerce': {
                'employees': (80, 150),
                'outlets': (30, 60),
                'revenue': (310000.3, 7965453.4)
            }
        }
    }

    city_data = generate_city(5, **city_layout)
    print(pf(city_data['analysis']))