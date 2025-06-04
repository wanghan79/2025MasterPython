import random
import string
import numpy as np
from datetime import datetime, timedelta
from functools import wraps

# 作业3：带参数的统计修饰器

def stat_metrics(*metrics):
    """
    统计修饰器，支持SUM/AVG/VAR/RMSE等任意组合
    Args:
        *metrics: 统计项（'SUM', 'AVG', 'VAR', 'RMSE'）
    Returns:
        修饰后的函数
    """
    def deco(func):
        @wraps(func)
        def inner(*args, **kwargs):
            res = func(*args, **kwargs)
            self = args[0]
            leaves = self.find_leaves(res)
            nums = [leaf['val'] for leaf in leaves if isinstance(leaf['val'], (int, float))]
            out = {}
            if nums:
                for m in metrics:
                    if m.upper() == 'AVG':
                        out['AVG'] = float(np.mean(nums))
                    elif m.upper() == 'VAR':
                        out['VAR'] = float(np.var(nums))
                    elif m.upper() == 'RMSE':
                        out['RMSE'] = float(np.sqrt(np.mean(np.square(nums))))
                    elif m.upper() == 'SUM':
                        out['SUM'] = float(np.sum(nums))
            return res, out
        return inner
    return deco

class DataGen:
    def __init__(self, rng=None):
        self.rng = rng or random
    def random_val(self, dtype, drange=None):
        if dtype == int:
            return self.rng.randint(drange[0], drange[1])
        elif dtype == float:
            return self.rng.uniform(drange[0], drange[1])
        elif dtype == str:
            if isinstance(drange, int):
                return ''.join(self.rng.choices(string.ascii_letters, k=drange))
            elif isinstance(drange, list):
                return self.rng.choice(drange)
        elif dtype == bool:
            return self.rng.choice([True, False])
        elif dtype == list:
            return [self.random_val(drange['type'], drange.get('range')) for _ in range(drange['length'])]
        elif dtype == tuple:
            return tuple(self.random_val(drange['type'], drange.get('range')) for _ in range(drange['length']))
        elif dtype == dict:
            return self.make_struct(drange)
        elif dtype == 'date':
            start, end = drange
            days = self.rng.randint(0, (end - start).days)
            return start + timedelta(days=days)
        else:
            return None
    def make_struct(self, struct):
        if isinstance(struct, dict):
            ret = {}
            for k, v in struct.items():
                if isinstance(v, dict):
                    tp = v.get('type')
                    rg = v.get('range')
                    sub = v.get('subs', [])
                    if sub:
                        ret[k] = [self.make_struct(s) for s in sub]
                    else:
                        ret[k] = self.random_val(tp, rg)
                else:
                    ret[k] = v
            return ret
        else:
            raise Exception('结构定义必须为dict')
    def find_leaves(self, data, path=""):
        leaves = []
        if isinstance(data, dict):
            for k, v in data.items():
                p = f"{path}.{k}" if path else k
                leaves.extend(self.find_leaves(v, p))
        elif isinstance(data, (list, tuple)):
            for i, v in enumerate(data):
                p = f"{path}[{i}]"
                leaves.extend(self.find_leaves(v, p))
        else:
            leaves.append({'path': path, 'val': data})
        return leaves
    @stat_metrics('AVG', 'VAR', 'RMSE', 'SUM')
    def analyze(self, data):
        return data

if __name__ == "__main__":
    struct = {
        'user_id': {'type': int, 'range': (1000, 9999)},
        'username': {'type': str, 'range': 8},
        'is_active': {'type': bool},
        'score': {'type': float, 'range': (0.0, 100.0)},
        'tags': {
            'type': list,
            'range': {
                'type': str,
                'range': 5,
                'length': 3
            }
        },
        'login_history': {
            'type': dict,
            'subs': [
                {'last_login': {'type': 'date', 'range': [datetime(2023, 1, 1), datetime(2023, 12, 31)]}},
                {'login_count': {'type': int, 'range': (0, 100)}}
            ]
        }
    }
    gen = DataGen()
    sample = gen.make_struct(struct)
    print("生成样本:")
    print(sample)
    print()
    _, metrics = gen.analyze(sample)
    print("统计指标:")
    print(metrics)
