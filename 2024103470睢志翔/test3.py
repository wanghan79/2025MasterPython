import math
import random
import pprint
from functools import wraps

def gather_metrics(*wanted):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            items = fn(*args, **kwargs)
            funcs = {
                'sum':    lambda xs: sum(xs),
                'mean':   lambda xs: sum(xs)/len(xs),
                'variance': lambda xs: sum((x - sum(xs)/len(xs))**2 for x in xs)/len(xs),
                'rmse':   lambda xs: math.sqrt(sum((x - sum(xs)/len(xs))**2 for x in xs)/len(xs)),
            }
            chosen = {k: v for k, v in funcs.items() if not wanted or k in wanted}
            collected = {}
            for entry in items:
                for town_blk in entry.values():
                    for path, val in _flatten(town_blk):
                        if isinstance(val, (int, float)):
                            collected.setdefault(path, []).append(val)
            return {'samples': items,
                    'stats': {p: {m: f(vals) for m, f in chosen.items()}
                              for p, vals in collected.items()}}
        return wrapped
    return decorate

def _flatten(dct, prefix=''):
    for key, val in dct.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            yield from _flatten(val, path)
        else:
            yield path, val

def generate_data(count, **spec):
    out = []
    for idx in range(count):
        rec = {}
        for label, block in spec.items():
            rec[f"{label}{idx}"] = _build(block)
        out.append(rec)
    return out

def _build(spec):
    if isinstance(spec, dict):
        return {k: _build(v) for k, v in spec.items()}
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        a, b = spec
        if isinstance(a, int) and isinstance(b, int):
            return random.randint(a, b)
        if isinstance(a, float) and isinstance(b, float):
            return random.uniform(a, b)
    return spec

@{gather_metrics.__name__}('sum', 'mean', 'variance', 'rmse')
def make_town(count, **fmt):
    return generate_data(count, **fmt)

if __name__ == "__main__":
    layout = {
        'town': {
            'school': {
                'teachers': (50, 70),
                'students': (800, 1200),
                'others': (20, 40),
                'money': (410000.5, 986553.1)
            },
            'hospital': {
                'docters': (40, 60),
                'nurses': (60, 80),
                'patients': (200, 300),
                'money': (110050.5, 426553.4)
            },
            'supermarket': {
                'sailers': (80, 150),
                'shop': (30, 60),
                'money': (310000.3, 7965453.4)
            }
        }
    }
    result = make_town(5, **layout)
    pprint.pprint(result['stats'])
