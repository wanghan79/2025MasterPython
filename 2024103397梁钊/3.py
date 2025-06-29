import numpy as np
from functools import wraps

def stats_analyzer(*stats):
    """
    统计分析装饰器，用于计算生成样本中所有数值型数据的统计特征
    
    参数:
        *stats: 统计项名称，支持 'SUM', 'AVG', 'VAR', 'RMSE'
        
    返回:
        function: 装饰后的函数，返回原始结果和统计分析结果
    """
    valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
    selected_stats = [stat.upper() for stat in stats if stat.upper() in valid_stats]
    
    if not selected_stats:
        raise ValueError("至少需要指定一个有效的统计项: SUM, AVG, VAR, RMSE")
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
    
            result = func(*args, **kwargs)
            
            # 递归提取所有数值型数据
            def extract_numeric(data):
                values = []
                if isinstance(data, (int, float, np.number)):
                    values.append(data)
                elif isinstance(data, (list, tuple, set, np.ndarray)):
                    for item in data:
                        values.extend(extract_numeric(item))
                elif isinstance(data, dict):
                    for v in data.values():
                        values.extend(extract_numeric(v))
                return values
            
            # 对每个样本集合进行统计分析
            stats_results = {}
            for key, samples in result.items():
                all_values = []
                for sample in samples:
                    all_values.extend(extract_numeric(sample))
                
                if not all_values:
                    stats_results[key] = {stat: 'N/A' for stat in selected_stats}
                    continue
                
                # 计算统计值
                stats_dict = {}
                if 'SUM' in selected_stats:
                    stats_dict['SUM'] = sum(all_values)
                if 'AVG' in selected_stats:
                    stats_dict['AVG'] = np.mean(all_values)
                if 'VAR' in selected_stats:
                    stats_dict['VAR'] = np.var(all_values)
                if 'RMSE' in selected_stats:
                    stats_dict['RMSE'] = np.sqrt(np.mean(np.square(all_values)))
                
                stats_results[key] = stats_dict
            
            return result, stats_results
        
        return wrapper
    return decorator

# 导入作业2中的函数
try:
    from random_data_generator import generate_random_samples
except ImportError:
    # 如果无法导入，则提供一个模拟函数用于测试
    def generate_random_samples(**kwargs):
        """模拟作业2中的函数，仅用于测试装饰器"""
        return {
            'list_int': [
                [1, 2, 3],
                [4, 5, 6]
            ],
            'dict_float': [
                {'a': 1.1, 'b': 2.2},
                {'a': 3.3, 'b': 4.4}
            ]
        }

# 应用装饰器到generate_random_samples函数
@stats_analyzer('SUM', 'AVG', 'VAR', 'RMSE')
def decorated_generate_random_samples(**kwargs):
    return generate_random_samples(**kwargs)

if __name__ == "__main__":
    # 示例用法
    samples, stats = decorated_generate_random_samples(
        list_of_ints={"list": {"count": 3, "element": {"int": {"min": 1, "max": 10}}}, "count": 2},
        dict_of_floats={"dict": {"keys": ["a", "b"], "values": {"float": {"min": 0.0, "max": 5.0}}}, "count": 2}
    )
    
    # 打印生成的样本
    print("\n生成的样本:")
    for key, value in samples.items():
        print(f"{key}: {value}")
    
    # 打印统计结果
    print("\n统计分析结果:")
    for key, value in stats.items():
        print(f"{key}: {value}")