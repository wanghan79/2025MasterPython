import numpy as np
from datetime import date
import random
import string


def statistics_decorator(*stats):
    def decorator(func):
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            stats_result = {}

            def analyze(samples):
                values = []
                for sample in samples:
                    for v in sample.values():
                        if type(v) == int or type(v) == float:
                            values.append(v)
                        elif isinstance(v, list):
                            v = [i for i in v if type(
                                i) == int or type(i) == float]
                            values.extend(v)
                        elif isinstance(v, dict):
                            values.extend(analyze([v]))
                return values

            values = analyze(samples)
            if 'SUM' in stats:
                stats_result['SUM'] = np.sum(values)
            if 'AVG' in stats:
                stats_result['AVG'] = np.mean(values)
            if 'VAR' in stats:
                stats_result['VAR'] = np.var(values)
            if 'RMSE' in stats:
                stats_result['RMSE'] = np.sqrt(
                    np.mean(np.square(np.array(values))))
            return samples, {'values ': values}, stats_result
        return wrapper
    return decorator


class DataSampler:
    def __init__(self, **kwargs):
        self.count = kwargs['count']
        self.structure = {k: v for k, v in kwargs.items() if k != 'count'}

    @statistics_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def __call__(self):
        def sample(item):
            if isinstance(item, dict):
                return {k: sample(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [sample(i) for i in item]
            elif item == int:
                return random.randint(0, 65535)
            elif item == float:
                return random.uniform(0, 7355608)
            elif item == date:
                return date.today().replace(day=random.randint(1, 28), month=random.randint(1, 12), year=random.randint(2000, 2077))
            elif item == str:
                length = random.randint(1, 10)
                return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            elif item == bool:
                return random.choice([True, False])
            else:
                return None
        return [sample(self.structure) for _ in range(self.count)]


if __name__ == "__main__":

    datasampler = DataSampler(
        count=2,
        id=int,
        name=str,
        user_info={'age': int, 'email': str},
        date=date,
        scores=[float]
    )
    res = datasampler()
    print(res)

# result:
# ([{'id': 8799, 'name': '3YlHqlTX', 'user_info': {'age': 65147, 'email': 'fjlz4uqgyz'}, 'date': datetime.date(2068, 8, 25), 'scores': [3705624.727709602]},
# {'id': 61331, 'name': 'rn5Wr7y', 'user_info': {'age': 62890, 'email': '45leq'}, 'date': datetime.date(2005, 1, 14), 'scores': [3958233.6505398387]}],
# {'values ': [8799, 65147, 3705624.727709602, 61331, 62890, 3958233.6505398387]},
# {'SUM': np.float64(7862025.37824944), 'AVG': np.float64(1310337.5630415734), 'VAR': np.float64(3184899879640.5703), 'RMSE': np.float64(2214019.965754216)})
