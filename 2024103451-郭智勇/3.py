import math

def stats_decorator(*stat_items):
    def decorator(func):
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            numeric_data = []
            for row in data:
                numeric_row = [x for x in row if isinstance(x, (int, float))]
                if numeric_row:
                    numeric_data.append(numeric_row)
            if not numeric_data:
                return {}
            cols = list(zip(*numeric_data))
            results = {}
            for idx, col in enumerate(cols):
                col_stats = {}
                n = len(col)
                s = sum(col)
                mean = s / n if n else 0
                var = sum((x - mean) ** 2 for x in col) / n if n else 0
                rmse = math.sqrt(sum((x - mean) ** 2 for x in col) / n) if n else 0
                for stat in stat_items:
                    if stat.upper() == 'SUM':
                        col_stats['SUM'] = s
                    if stat.upper() == 'AVG':
                        col_stats['AVG'] = mean
                    if stat.upper() == 'VAR':
                        col_stats['VAR'] = var
                    if stat.upper() == 'RMSE':
                        col_stats['RMSE'] = rmse
                results[f'col_{idx}'] = col_stats
            return results
        return wrapper
    return decorator

def get_sample_data():
    return [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

@stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def get_numeric_stats():
    return get_sample_data()

stats = get_numeric_stats()
print(stats)