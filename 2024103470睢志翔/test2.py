# coding=utf-8
import random

def DataGenerate(num, **kwargs):
    result = []
    for i in range(num):
        element = {}
        for k, v in kwargs.items():
            element[f"{k}{i}"] = _gen(v)
        result.append(element)
    return result

def _gen(x):
    if isinstance(x, dict):
        return {k: _gen(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)) and len(x) == 2:
        a, b = x
        if isinstance(a, int) and isinstance(b, int):
            return random.randint(a, b)
        if isinstance(a, float) and isinstance(b, float):
            return random.uniform(a, b)
    return x

if __name__ == '__main__':
    fmt = {
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
    data = DataGenerate(5, **fmt)
    for e in data:
        print(e)
