import math
import collections

def generate_random_value(data_type, depth, max_depth, list_avg_len=3, dict_avg_len=3):
    if data_type == 'int':
        return random.randint(0, 100)
    elif data_type == 'float':
        return round(random.uniform(0.0, 100.0), 2)
    elif data_type == 'str':
        length = random.randint(5, 15)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    elif data_type == 'bool':
        return random.choice([True, False])
    elif data_type == 'list':
        if depth >= max_depth:
            return [generate_random_value(random.choice(['int', 'float', 'str', 'bool']), depth + 1, max_depth)
                    for _ in range(random.randint(1, list_avg_len * 2))]
        else:
            possible_types = ['int', 'float', 'str', 'bool', 'list', 'dict']
            return [generate_random_value(random.choice(possible_types), depth + 1, max_depth, list_avg_len, dict_avg_len)
                    for _ in range(random.randint(1, list_avg_len * 2))]
    elif data_type == 'dict':
        if depth >= max_depth:
            return {
                f"key_{i}": generate_random_value(random.choice(['int', 'float', 'str', 'bool']), depth + 1, max_depth)
                for i in range(random.randint(1, dict_avg_len * 2))
            }
        else:
            possible_types = ['int', 'float', 'str', 'bool', 'list', 'dict']
            return {
                f"key_{i}": generate_random_value(random.choice(possible_types), depth + 1, max_depth, list_avg_len, dict_avg_len)
                for i in range(random.randint(1, dict_avg_len * 2))
            }
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")

def generate_nested_sample_set(**kwargs):
    sample_count = kwargs.get('sample_count')
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("`sample_count` 必须是大于0的整数。")

    max_depth = kwargs.get('max_depth', 3)
    root_type = kwargs.get('root_type', 'dict')
    list_avg_len = kwargs.get('list_avg_len', 3)
    dict_avg_len = kwargs.get('dict_avg_len', 3)
    possible_leaf_types = kwargs.get('possible_leaf_types', ['int', 'float', 'str', 'bool'])

    if root_type not in ['list', 'dict']:
        raise ValueError("`root_type` 必须是 'list' 或 'dict'。")

    samples = []
    for _ in range(sample_count):
        if root_type == 'list':
            sample = [generate_random_value(random.choice(possible_leaf_types + ['list', 'dict']), 
                                            1, max_depth, list_avg_len, dict_avg_len) 
                      for _ in range(random.randint(1, list_avg_len * 2))]
        else: # root_type == 'dict'
            sample = {
                f"root_key_{i}": generate_random_value(random.choice(possible_leaf_types + ['list', 'dict']), 
                                                      1, max_depth, list_avg_len, dict_avg_len)
                for i in range(random.randint(1, dict_avg_len * 2))
            }
        samples.append(sample)
    return samples


def collect_numbers(data):
    numbers = []
    if isinstance(data, (int, float)):
        numbers.append(data)
    elif isinstance(data, list):
        for item in data:
            numbers.extend(collect_numbers(item))
    elif isinstance(data, dict):
        for key, value in data.items():
            numbers.extend(collect_numbers(value))
    return numbers

class DataStatsDecorator:
    def __init__(self, *args, **kwargs):
        self.stats_to_compute = set()
        
        for arg in args:
            if isinstance(arg, str):
                self.stats_to_compute.add(arg.upper())

        valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
        for stat, enabled in kwargs.items():
            if stat.upper() in valid_stats and enabled:
                self.stats_to_compute.add(stat.upper())
            elif stat.upper() in valid_stats and not enabled:
                self.stats_to_compute.discard(stat.upper())

        if not self.stats_to_compute:
            print("警告: 未指定任何统计项，修饰器将不执行任何统计。")

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            sample_set = func(*args, **kwargs)

            all_numbers = []
            for sample in sample_set:
                all_numbers.extend(collect_numbers(sample))

            if not all_numbers:
                print("未找到任何数值型数据进行统计。")
                return sample_set # 返回原始样本集

            results = {}
            n = len(all_numbers)

            # 统计 SUM (求和)
            if 'SUM' in self.stats_to_compute:
                total_sum = sum(all_numbers)
                results['SUM'] = total_sum

            # 统计 AVG (均值)
            if 'AVG' in self.stats_to_compute:
                if n > 0:
                    average = sum(all_numbers) / n
                    results['AVG'] = average
                else:
                    results['AVG'] = float('nan') # 如果没有数据，均值为NaN

            # 统计 VAR (方差)
            if 'VAR' in self.stats_to_compute:
                if n > 1: # 方差至少需要2个数据点
                    # 计算均值
                    if 'AVG' in results:
                        mean_val = results['AVG']
                    else:
                        mean_val = sum(all_numbers) / n if n > 0 else 0

                    variance = sum((x - mean_val) ** 2 for x in all_numbers) / (n - 1)
                    results['VAR'] = variance
                else:
                    results['VAR'] = float('nan') # 如果数据点不足，方差为NaN

            if 'RMSE' in self.stats_to_compute:
                if n > 0:
                    if 'AVG' in results:
                        mean_val = results['AVG']
                    else:
                        mean_val = sum(all_numbers) / n if n > 0 else 0

                    mse = sum((x - mean_val) ** 2 for x in all_numbers) / n
                    rmse = math.sqrt(mse)
                    results['RMSE'] = rmse
                else:
                    results['RMSE'] = float('nan') # 如果没有数据，RMSE为NaN

            print("\n--- 数值数据统计结果 ---")
            for stat, value in results.items():
                print(f"{stat}: {value:.4f}")
            print("------------------------")

            return sample_set # 返回原始样本集

        return wrapper

if __name__ == "__main__":
    print("--- 示例 1: 统计 SUM 和 AVG ---")
    @DataStatsDecorator('SUM', avg=True)
    def decorated_generator_1(**kwargs):
        return generate_nested_sample_set(**kwargs)

    sample_data1 = decorated_generator_1(sample_count=2, max_depth=2, root_type='list')
    print("生成的样本数据 (1):")
    for s in sample_data1:
        print(s)

    print("\n--- 示例 2: 统计所有项 ---")
    @DataStatsDecorator(sum=True, avg=True, var=True, rmse=True)
    def decorated_generator_2(**kwargs):
        return generate_nested_sample_set(**kwargs)

    sample_data2 = decorated_generator_2(sample_count=3, max_depth=3, root_type='dict',
                                        possible_leaf_types=['int', 'float'])
    print("生成的样本数据 (2):")
    for s in sample_data2:
        print(s)

    print("\n--- 示例 3: 只统计 VAR 和 RMSE ---")
    @DataStatsDecorator('VAR', 'RMSE')
    def decorated_generator_3(**kwargs):
        return generate_nested_sample_set(**kwargs)

    sample_data3 = decorated_generator_3(sample_count=1, max_depth=1, root_type='list',
                                        possible_leaf_types=['int'])
    print("生成的样本数据 (3):")
    for s in sample_data3:
        print(s)

    print("\n--- 示例 4: 没有数值数据的情况 ---")
    @DataStatsDecorator(sum=True, avg=True)
    def decorated_generator_4(**kwargs):
        return generate_nested_sample_set(**kwargs)
    
    sample_data4 = decorated_generator_4(sample_count=1, max_depth=1, root_type='dict',
                                        possible_leaf_types=['str', 'bool'])
    print("生成的样本数据 (4):")
    for s in sample_data4:
        print(s)

    print("\n--- 示例 5: 没有指定统计项的情况 ---")
    @DataStatsDecorator()
    def decorated_generator_5(**kwargs):
        return generate_nested_sample_set(**kwargs)
    
    sample_data5 = decorated_generator_5(sample_count=1, max_depth=1)
    print("生成的样本数据 (5):")
    for s in sample_data5:
        print(s)
