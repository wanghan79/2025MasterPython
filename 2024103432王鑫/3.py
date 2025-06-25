import random
import string
from datetime import datetime, timedelta
import math

def generate_random_data(**kwargs):
    def _get_random_value(data_spec):
        data_type = data_spec['type']
        data_range = data_spec.get('range')

        if data_type == int:
            return random.randint(data_range[0], data_range[1])
        elif data_type == float:
            return random.uniform(data_range[0], data_range[1])
        elif data_type == str:
            length = data_range
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        elif data_type == bool:
            return random.choice([True, False])
        elif data_type == list:
            length = data_spec.get('length', 1)
            item_spec = data_range
            return [_get_random_value(item_spec) for _ in range(length)]
        elif data_type == tuple:
            length = data_spec.get('length', 1)
            item_spec = data_range
            return tuple(_get_random_value(item_spec) for _ in range(length))
        elif data_type == dict:
            return _generate_nested_structure(data_range)
        elif data_type == 'date':
            start_date, end_date = data_range
            time_delta = end_date - start_date
            random_days = random.randint(0, time_delta.days)
            return start_date + timedelta(days=random_days)
        else:
            return None

    def _generate_nested_structure(structure_spec):
        if isinstance(structure_spec, dict):
            if 'type' in structure_spec:
                return _get_random_value(structure_spec)
            
            generated_data = {}
            for key, value_spec in structure_spec.items():
                generated_data[key] = _generate_nested_structure(value_spec)
            return generated_data
        elif isinstance(structure_spec, list):
            return [_generate_nested_structure(item_spec) for item_spec in structure_spec]
        elif isinstance(structure_spec, tuple):
            return tuple(_generate_nested_structure(item_spec) for item_spec in structure_spec)
        else:
            raise ValueError("Unsupported structure specification type")

    num_samples = kwargs.get('num_samples', 1)
    structure_template = kwargs.get('structure')

    if not structure_template:
        raise ValueError("'structure' argument is required and must define the data structure.")

    samples = []
    for _ in range(num_samples):
        samples.append(_generate_nested_structure(structure_template))
    return samples


def stats_decorator(*enabled_stats):
    def decorator(func):
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs) # 调用被修饰的函数生成样本
            
            numeric_values = []

            # 递归函数，用于从嵌套结构中提取所有数值型叶节点
            def extract_numeric_values(data):
                if isinstance(data, (int, float)):
                    numeric_values.append(data)
                elif isinstance(data, (list, tuple)):
                    for item in data:
                        extract_numeric_values(item)
                elif isinstance(data, dict):
                    for value in data.values():
                        extract_numeric_values(value)

            for sample in samples:
                extract_numeric_values(sample)
            
            results = {}
            if not numeric_values:
                print("Warning: No numeric values found for statistics.")
                return samples, results

            # 计算统计量
            if 'SUM' in enabled_stats:
                results['SUM'] = sum(numeric_values)
            
            if 'AVG' in enabled_stats:
                results['AVG'] = sum(numeric_values) / len(numeric_values)
            
            if 'VAR' in enabled_stats:
                if len(numeric_values) > 1:
                    mean = results.get('AVG', sum(numeric_values) / len(numeric_values)) # 避免重复计算均值
                    variance = sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1)
                    results['VAR'] = variance
                else:
                    results['VAR'] = 0.0 # 单个数据点的方差为0
            
            if 'RMSE' in enabled_stats:
                if 'VAR' in results:
                    results['RMSE'] = math.sqrt(results['VAR']) # RMSE在此处等同于标准差 (Std Dev)
                else: # 如果VAR没有被计算，但请求了RMSE，则先计算VAR
                    if len(numeric_values) > 1:
                        mean = results.get('AVG', sum(numeric_values) / len(numeric_values))
                        variance = sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1)
                        results['RMSE'] = math.sqrt(variance)
                    else:
                        results['RMSE'] = 0.0

            return samples, results
        return wrapper
    return decorator

# ---
if __name__ == '__main__':
    # 定义数据结构模板
    user_data_template = {
        'user_id': {'type': int, 'range': (1000, 9999)},
        'username': {'type': str, 'range': 10},
        'is_active': {'type': bool},
        'profile': {
            'email': {'type': str, 'range': 15},
            'age': {'type': int, 'range': (18, 90)},
            'scores': { # 嵌套列表，包含数值
                'type': list,
                'range': {'type': float, 'range': (60.0, 100.0)},
                'length': random.randint(3, 5)
            },
            'last_login': {'type': 'date', 'range': (datetime(2023, 1, 1), datetime.now())}
        },
        'settings': {
            'notifications': {'type': bool},
            'preferences': {
                'theme': {'type': str, 'range': 7},
                'language_codes': {'type': tuple, 'range': {'type': str, 'range': 2}, 'length': 2}
            }
        },
        'transaction_history': {
            'type': list,
            'length': 3,
            'range': { 
                'type': dict,
                'range': {
                    'transaction_id': {'type': str, 'range': 8},
                    'amount': {'type': float, 'range': (10.0, 500.0)}, 
                    'quantity': {'type': int, 'range': (1, 5)}, 
                    'timestamp': {'type': 'date', 'range': (datetime(2024, 1, 1), datetime.now())}
                }
            }
        }
    }

    print("--- 仅计算求和与均值 ---")
    @stats_decorator('SUM', 'AVG')
    def get_user_data_with_sum_avg(num_samples, structure):
        return generate_random_data(num_samples=num_samples, structure=structure)

    samples_sum_avg, stats_sum_avg = get_user_data_with_sum_avg(num_samples=5, structure=user_data_template)
    # 打印前两个样本，以便查看生成的数据结构
    for i, sample in enumerate(samples_sum_avg[:2]):
        print(f"Sample {i+1}:")
        import json
        print(json.dumps(sample, indent=4, default=str))
    print("统计结果:", stats_sum_avg)

    print("\n--- 计算所有统计项 ---")
    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def get_user_data_all_stats(num_samples, structure):
        return generate_random_data(num_samples=num_samples, structure=structure)

    samples_all_stats, stats_all_stats = get_user_data_all_stats(num_samples=5, structure=user_data_template)
    print("统计结果:", stats_all_stats)

    print("\n--- 仅计算方差和均方根误差 ---")
    @stats_decorator('VAR', 'RMSE')
    def get_user_data_var_rmse(num_samples, structure):
        return generate_random_data(num_samples=num_samples, structure=structure)
    
    samples_var_rmse, stats_var_rmse = get_user_data_var_rmse(num_samples=5, structure=user_data_template)
    print("统计结果:", stats_var_rmse)

    print("\n--- 没有数值数据的情况 ---")
    no_numeric_template = {
        'name': {'type': str, 'range': 5},
        'id': {'type': str, 'range': 3}
    }

    @stats_decorator('SUM', 'AVG')
    def get_no_numeric_data(num_samples, structure):
        return generate_random_data(num_samples=num_samples, structure=structure)
    
    samples_no_numeric, stats_no_numeric = get_no_numeric_data(num_samples=2, structure=no_numeric_template)
    print("统计结果:", stats_no_numeric)