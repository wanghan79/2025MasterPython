# 作业3：Python修饰器编写
# 要求：编写带参数的修饰器，修饰作业2封装的函数，
# 实现对其生成样本集中所有数值型数据的统计操作
# 统计项包括：SUM（求和）、AVG（均值）、VAR（方差）、RMSE（均方根差）

import math
import functools
from typing import Any, List, Dict, Union, Callable
import sys
import os

# 导入作业2的函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../作业2')
from assignment2 import generate_nested_samples

class StatisticsDecorator:
    """
    带参数的统计修饰器类
    支持对数值型数据进行SUM、AVG、VAR、RMSE统计
    """
    
    def __init__(self, *stats_types):
        """
        初始化修饰器
        
        参数:
            *stats_types: 要计算的统计类型，可选值：'SUM', 'AVG', 'VAR', 'RMSE'
        """
        valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
        self.stats_types = []
        
        for stat in stats_types:
            if stat.upper() in valid_stats:
                self.stats_types.append(stat.upper())
            else:
                raise ValueError(f"不支持的统计类型: {stat}. 支持的类型: {valid_stats}")
        
        if not self.stats_types:
            self.stats_types = ['SUM', 'AVG', 'VAR', 'RMSE']  # 默认全部统计
        
        print(f"统计修饰器初始化，将计算: {', '.join(self.stats_types)}")
    
    def __call__(self, func: Callable) -> Callable:
        """
        修饰器调用方法
        
        参数:
            func: 被修饰的函数
        
        返回:
            Callable: 修饰后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原函数获取样本数据
            print(f"调用原函数: {func.__name__}")
            samples = func(*args, **kwargs)
            
            # 提取所有数值型数据
            numeric_values = self._extract_numeric_values(samples)
            print(f"提取到 {len(numeric_values)} 个数值型数据")
            
            # 计算统计信息
            statistics = self._calculate_statistics(numeric_values)
            
            # 输出统计结果
            self._print_statistics(statistics)
            
            # 返回原始样本数据和统计结果
            return {
                'samples': samples,
                'statistics': statistics,
                'numeric_count': len(numeric_values)
            }
        
        return wrapper
    
    def _extract_numeric_values(self, data: Any) -> List[Union[int, float]]:
        """
        递归提取数据结构中的所有数值型数据
        
        参数:
            data: 要提取的数据结构
        
        返回:
            List[Union[int, float]]: 提取到的数值列表
        """
        numeric_values = []
        
        if isinstance(data, (int, float)):
            numeric_values.append(data)
        
        elif isinstance(data, (list, tuple)):
            for item in data:
                numeric_values.extend(self._extract_numeric_values(item))
        
        elif isinstance(data, dict):
            for value in data.values():
                numeric_values.extend(self._extract_numeric_values(value))
        
        return numeric_values
    
    def _calculate_statistics(self, values: List[Union[int, float]]) -> Dict[str, float]:
        """
        计算统计信息
        
        参数:
            values: 数值列表
        
        返回:
            Dict[str, float]: 统计结果字典
        """
        if not values:
            return {stat: 0.0 for stat in self.stats_types}
        
        statistics = {}
        n = len(values)
        
        # 计算求和 (SUM)
        if 'SUM' in self.stats_types:
            statistics['SUM'] = sum(values)
        
        # 计算均值 (AVG)
        if 'AVG' in self.stats_types:
            statistics['AVG'] = sum(values) / n
        
        # 计算方差 (VAR)
        if 'VAR' in self.stats_types:
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / n
            statistics['VAR'] = variance
        
        # 计算均方根差 (RMSE)
        if 'RMSE' in self.stats_types:
            mean = sum(values) / n
            mse = sum((x - mean) ** 2 for x in values) / n
            statistics['RMSE'] = math.sqrt(mse)
        
        return statistics
    
    def _print_statistics(self, statistics: Dict[str, float]):
        """
        打印统计结果
        
        参数:
            statistics: 统计结果字典
        """
        print("\n" + "=" * 50)
        print("数值型数据统计结果")
        print("=" * 50)
        
        for stat_type, value in statistics.items():
            if stat_type == 'SUM':
                print(f"求和 (SUM):     {value:.4f}")
            elif stat_type == 'AVG':
                print(f"均值 (AVG):     {value:.4f}")
            elif stat_type == 'VAR':
                print(f"方差 (VAR):     {value:.4f}")
            elif stat_type == 'RMSE':
                print(f"均方根差 (RMSE): {value:.4f}")
        
        print("=" * 50)

# 函数形式的修饰器
def statistics_decorator(*stats_types):
    """
    函数形式的统计修饰器
    
    参数:
        *stats_types: 要计算的统计类型
    
    返回:
        修饰器函数
    """
    def decorator(func):
        # 创建统计修饰器实例
        stats_decorator = StatisticsDecorator(*stats_types)
        return stats_decorator(func)
    
    return decorator

# 演示函数
@StatisticsDecorator('SUM', 'AVG')
def demo_partial_stats(**kwargs):
    """演示部分统计功能的修饰器"""
    return generate_nested_samples(**kwargs)

@StatisticsDecorator()  # 使用默认的全部统计
def demo_full_stats(**kwargs):
    """演示全部统计功能的修饰器"""
    return generate_nested_samples(**kwargs)

@statistics_decorator('VAR', 'RMSE')
def demo_variance_stats(**kwargs):
    """演示方差相关统计的修饰器"""
    return generate_nested_samples(**kwargs)

def test_simple_data():
    """测试简单数据类型的统计"""
    print("测试1: 简单数值数据统计")
    print("-" * 40)
    
    result = demo_partial_stats(
        sample_count=10,
        structure={'type': 'int', 'min': 1, 'max': 100}
    )
    
    print(f"生成的样本: {result['samples']}")
    print(f"数值个数: {result['numeric_count']}")

def test_nested_data():
    """测试嵌套数据类型的统计"""
    print("\n测试2: 嵌套数据类型统计")
    print("-" * 40)
    
    result = demo_full_stats(
        sample_count=3,
        structure={
            'type': 'dict',
            'keys': {
                'scores': {
                    'type': 'list',
                    'element_type': {'type': 'float', 'min': 60.0, 'max': 100.0},
                    'length': 4
                },
                'age': {'type': 'int', 'min': 18, 'max': 25},
                'name': {'type': 'str', 'length': 6}
            }
        }
    )
    
    print(f"数值个数: {result['numeric_count']}")

def test_complex_nested_data():
    """测试复杂嵌套数据类型的统计"""
    print("\n测试3: 复杂嵌套数据类型统计")
    print("-" * 40)
    
    result = demo_variance_stats(
        sample_count=2,
        structure={
            'type': 'list',
            'element_type': {
                'type': 'dict',
                'keys': {
                    'measurements': {
                        'type': 'tuple',
                        'elements': [
                            {'type': 'float', 'min': 0.0, 'max': 10.0},
                            {'type': 'float', 'min': 0.0, 'max': 10.0},
                            {'type': 'int', 'min': 1, 'max': 100}
                        ]
                    },
                    'metadata': {
                        'type': 'dict',
                        'keys': {
                            'temperature': {'type': 'float', 'min': -10.0, 'max': 40.0},
                            'humidity': {'type': 'int', 'min': 0, 'max': 100}
                        }
                    }
                }
            },
            'length': 3
        }
    )
    
    print(f"数值个数: {result['numeric_count']}")

def main():
    """主函数：运行所有测试"""
    print("Python修饰器统计功能演示")
    print("=" * 60)
    
    # 运行测试
    test_simple_data()
    test_nested_data()
    test_complex_nested_data()
    
    print("\n" + "=" * 60)
    print("修饰器演示完成！")
    print("=" * 60)
    
    print("\n功能说明:")
    print("1. 支持类修饰器和函数修饰器两种形式")
    print("2. 可以选择性计算SUM、AVG、VAR、RMSE统计项")
    print("3. 自动递归提取嵌套数据结构中的所有数值")
    print("4. 返回原始样本数据和统计结果")

if __name__ == "__main__":
    main()
