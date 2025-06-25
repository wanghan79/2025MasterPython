import numpy as np
from work2 import DataSampler  # 引用作业2中的数据生成函数

def stats_decorator(*stats):
    """
    统计分析装饰器，用于计算数据样本的统计特征
    
    参数:
        stats: 需要计算的统计项，可选值为 'SUM', 'AVG', 'VAR', 'RMSE'
    """
    valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
    invalid_stats = [stat for stat in stats if stat not in valid_stats]
    
    if invalid_stats:
        raise ValueError(f"Invalid stats: {', '.join(invalid_stats)}. Valid options are {', '.join(valid_stats)}")
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            analyzer = DataAnalyzer(samples)
            results = analyzer.analyze(stats)
            return samples, results
        return wrapper
    return decorator

class DataAnalyzer:
    """数据统计分析器"""
    
    def __init__(self, samples):
        self.samples = samples
        
    def analyze(self, stats):
        """
        分析数据样本，计算指定的统计特征
        
        参数:
            stats: 需要计算的统计项
            
        返回:
            包含所有数值型字段统计结果的字典
        """
        # 收集所有数值型叶节点数据
        numerical_data = self._collect_numerical_data()
        
        # 计算统计特征
        results = {}
        for field, values in numerical_data.items():
            field_stats = {}
            if 'SUM' in stats:
                field_stats['SUM'] = sum(values)
            if 'AVG' in stats:
                field_stats['AVG'] = np.mean(values)
            if 'VAR' in stats:
                field_stats['VAR'] = np.var(values)
            if 'RMSE' in stats:
                field_stats['RMSE'] = np.sqrt(np.mean(np.square(values)))
            results[field] = field_stats
            
        return results
    
    def _collect_numerical_data(self):
        """收集所有数值型叶节点的数据"""
        numerical_data = {}
        
        for sample in self.samples:
            self._extract_numerical_values(sample, [], numerical_data)
            
        return numerical_data
    
    def _extract_numerical_values(self, obj, path, numerical_data):
        """递归提取数值型叶节点的值"""
        if isinstance(obj, (int, float)):
            # 数值型叶节点
            key = ".".join(path)
            if key not in numerical_data:
                numerical_data[key] = []
            numerical_data[key].append(obj)
        elif isinstance(obj, dict):
            # 字典类型
            for k, v in obj.items():
                # 跳过无统计意义的字段
                if k == 'street_number':
                    continue
                self._extract_numerical_values(v, path + [k], numerical_data)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            # 列表或元组类型
            for i, v in enumerate(obj):
                self._extract_numerical_values(v, path + [str(i)], numerical_data)
        # 其他类型（如字符串、日期）不处理

# 示例用法
if __name__ == "__main__":
    # 定义用户数据结构
    user_structure = {
        'id': {'__type__': int},
        'age': {'__type__': int, 'min': 18, 'max': 60},
        'height': {'__type__': float, 'min': 1.5, 'max': 2.0, 'precision': 2},
        'scores': {
            '__type__': list,
            '__element_type__': {'__type__': float, 'min': 0, 'max': 100},
            'min_length': 1,  # 确保至少有1个分数
            'max_length': 5   # 最多5个分数
        },
        'address': {
            '__type__': dict,
            '__structure__': {
                'city': {'__type__': str},
                'zip_code': {'__type__': str, 'length': 5}
            }
        }
    }
    
    # 使用装饰器修饰DataSampler函数，计算所有数值字段的统计信息
    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_user_data():
        return DataSampler(n_samples=3, structure=user_structure)
    
    # 生成数据并计算统计信息
    samples, stats = generate_user_data()
    
    # 输出结果
    print("生成的样本数据:")
    for i, sample in enumerate(samples):
        print(f"样本 {i+1}: {sample}")
    
    print("\n统计分析结果:")
    for field, stat_values in stats.items():
        print(f"字段: {field}")
        for stat, value in stat_values.items():
            print(f"  {stat}: {value:.2f}")
        print()