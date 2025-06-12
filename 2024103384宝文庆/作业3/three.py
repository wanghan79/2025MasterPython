import random
import numpy as np

def generate_random_sample(**kwargs):
    def generate_sample(levels):
        if levels == 0:
            return random.randint(0, 100)
        else:
            return [generate_sample(levels - 1) for _ in range(random.randint(1, 5))]

    sample_structure = kwargs.get("structure", 2)
    sample_count = kwargs.get("count", 1)
    return [generate_sample(sample_structure) for _ in range(sample_count)]


def statistics_decorator(*stats):
    def decorator(func):
        def wrapper(*args, **kwargs):

            samples = func(*args, **kwargs)
            stats_result = {}


            def flatten(data):
                if isinstance(data, list):
                    return [item for sublist in data for item in flatten(sublist)]
                else:
                    return [data]

            flattened_samples = flatten(samples)


            if 'SUM' in stats:
                stats_result['SUM'] = np.sum(flattened_samples)
            if 'AVG' in stats:
                stats_result['AVG'] = np.mean(flattened_samples)
            if 'VAR' in stats:
                stats_result['VAR'] = np.var(flattened_samples)
            if 'RMSE' in stats:
                stats_result['RMSE'] = np.sqrt(np.mean(np.square(np.array(flattened_samples))))

            return stats_result

        return wrapper

    return decorator


@statistics_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def generate_samples_for_statistics(**kwargs):
    return generate_random_sample(**kwargs)


statistics = generate_samples_for_statistics(structure=3, count=5)
print(statistics)
