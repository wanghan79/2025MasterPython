import functools
import numpy as np
from typing import List, Dict, Any, Callable
from assignment2.data_sampler import DataSampler

class StatsDecorator:
    def __init__(self, stats: List[str] = None):
        """
        初始化统计装饰器
        
        参数:
        stats: 需要计算的统计项列表，可选值：['SUM', 'AVG', 'VAR', 'RMSE']
        """
        self.stats = stats or ['SUM', 'AVG', 'VAR', 'RMSE']
        self.valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
        
        # 验证统计项
        invalid_stats = set(self.stats) - self.valid_stats
        if invalid_stats:
            raise ValueError(f"不支持的统计项: {invalid_stats}")
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取原始数据
            samples = func(*args, **kwargs)
            
            # 收集所有数值型数据
            numeric_values = self._collect_numeric_values(samples)
            
            # 计算统计结果
            results = {}
            for stat in self.stats:
                if stat == 'SUM':
                    results['SUM'] = sum(numeric_values)
                elif stat == 'AVG':
                    results['AVG'] = np.mean(numeric_values)
                elif stat == 'VAR':
                    results['VAR'] = np.var(numeric_values)
                elif stat == 'RMSE':
                    results['RMSE'] = np.sqrt(np.mean(np.square(numeric_values)))
            
            return {
                'samples': samples,
                'statistics': results
            }
        return wrapper
    
    def _collect_numeric_values(self, data: Any) -> List[float]:
        """
        递归收集所有数值型数据
        
        参数:
        data: 任意嵌套的数据结构
        
        返回:
        所有数值型数据的列表
        """
        numeric_values = []
        
        if isinstance(data, (int, float)):
            numeric_values.append(float(data))
        elif isinstance(data, (list, tuple)):
            for item in data:
                numeric_values.extend(self._collect_numeric_values(item))
        elif isinstance(data, dict):
            for value in data.values():
                numeric_values.extend(self._collect_numeric_values(value))
        
        return numeric_values

# 使用示例
if __name__ == "__main__":
    # 定义数据结构
    test_structure = {
        "id": int,
        "scores": [float, float, float],
        "nested": {
            "value": int,
            "array": [float, float]
        }
    }
    
    # 使用装饰器
    @StatsDecorator(['SUM', 'AVG', 'VAR', 'RMSE'])
    def generate_test_data(structure, num_samples):
        return DataSampler.generate_samples(structure, num_samples)
    
    # 生成数据并计算统计值
    result = generate_test_data(test_structure, 3)
    
    print("生成的样本：")
    for sample in result['samples']:
        print(sample)
    
    print("\n统计结果：")
    for stat_name, stat_value in result['statistics'].items():
        print(f"{stat_name}: {stat_value:.2f}") 