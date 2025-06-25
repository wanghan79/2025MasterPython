import random
import string
from datetime import datetime, timedelta

class RandomStructFactory:
    def __init__(self, rng=None):
        """
        随机结构体工厂
        Args:
            rng: 可选自定义随机数生成器
        """
        self.rng = rng or random
    
    def gen_value(self, dtype, drange=None):
        """
        按类型生成随机值
        Args:
            dtype: 数据类型
            drange: 数据范围
        Returns:
            随机值
        """
        if dtype == int:
            return self.rng.randint(drange[0], drange[1])
        elif dtype == float:
            return self.rng.uniform(drange[0], drange[1])
        elif dtype == str:
            if isinstance(drange, int):
                return ''.join(self.rng.choices(string.ascii_lowercase, k=drange))
            elif isinstance(drange, list):
                return self.rng.choice(drange)
        elif dtype == bool:
            return self.rng.choice([True, False])
        elif dtype == list:
            return [self.gen_value(drange['type'], drange.get('range')) for _ in range(drange['length'])]
        elif dtype == tuple:
            return tuple(self.gen_value(drange['type'], drange.get('range')) for _ in range(drange['length']))
        elif dtype == dict:
            return self.make_struct(drange)
        elif dtype == 'date':
            start, end = drange
            days = self.rng.randint(0, (end - start).days)
            return start + timedelta(days=days)
        else:
            return None
    
    def make_struct(self, struct_def):
        """
        递归生成嵌套结构体
        Args:
            struct_def: 结构定义
        Returns:
            结构体实例
        """
        if isinstance(struct_def, dict):
            out = {}
            for k, v in struct_def.items():
                if isinstance(v, dict):
                    tp = v.get('type')
                    rg = v.get('range')
                    sub = v.get('subs', [])
                    if sub:
                        out[k] = [self.make_struct(s) for s in sub]
                    else:
                        out[k] = self.gen_value(tp, rg)
                else:
                    out[k] = v
            return out
        else:
            raise Exception("结构定义必须为dict")
    
    def batch_samples(self, struct_def, count=1):
        """
        批量生成样本
        Args:
            struct_def: 结构定义
            count: 样本数
        Returns:
            样本列表
        """
        return [self.make_struct(struct_def) for _ in range(count)]

if __name__ == "__main__":
    # 示例结构
    struct = {
        'uid': {'type': int, 'range': (1000, 9999)},
        'name': {'type': str, 'range': 8},
        'active': {'type': bool},
        'score': {'type': float, 'range': (0.0, 100.0)},
        'labels': {
            'type': list,
            'range': {
                'type': str,
                'range': 5,
                'length': 3
            }
        },
        'history': {
            'type': dict,
            'subs': [
                {'last': {'type': 'date', 'range': [datetime(2023, 1, 1), datetime(2023, 12, 31)]}},
                {'count': {'type': int, 'range': (0, 100)}}
            ]
        }
    }
    factory = RandomStructFactory()
    samples = factory.batch_samples(struct, count=3)
    for idx, s in enumerate(samples, 1):
        print(f"样本{idx}:")
        print(s)
        print()
