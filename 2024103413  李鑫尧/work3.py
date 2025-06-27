import math
from functools import wraps
import random

# 带参数的装饰器实现
def stats_decorator(stats_list):
    """
    带参数的装饰器，用于计算样本集数值数据的统计指标
    :param stats_list: 统计项列表，可选值：'SUM', 'AVG', 'VAR', 'RMSE'
    :return: 装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取样本数据
            data = func(*args, **kwargs)
            
            # 提取数值型数据
            numeric_data = [x for x in data if isinstance(x, (int, float))]
            n = len(numeric_data)
            
            # 初始化结果字典
            results = {}
            
            if n == 0:
                # 如果没有数值数据，所有统计项设为0
                for stat in stats_list:
                    results[stat] = 0.0
                return results
            
            # 计算基础统计量
            total = sum(numeric_data)
            mean = total / n
            
            # 按需计算统计项
            if 'SUM' in stats_list:
                results['SUM'] = total
                
            if 'AVG' in stats_list:
                results['AVG'] = mean
                
            if 'VAR' in stats_list or 'RMSE' in stats_list:
                # 计算平方差总和
                squared_diff = sum((x - mean) ** 2 for x in numeric_data)
                
                if 'VAR' in stats_list:
                    # 样本方差（n-1）
                    results['VAR'] = squared_diff / (n - 1) if n > 1 else 0.0
                    
                if 'RMSE' in stats_list:
                    # 均方根差（总体标准差）
                    results['RMSE'] = math.sqrt(squared_diff / n)
            
            return results
        return wrapper
    return decorator

# 示例1：使用装饰器修饰生成样本集的函数
@stats_decorator(['SUM', 'AVG', 'VAR', 'RMSE'])
def generate_sample_1():
    """生成包含数值型和非数值型数据的样本集"""
    return [1.2, 3.5, 2.8, 4.1, 5.6, 'text', None]

# 示例2：只计算部分统计项
@stats_decorator(['AVG', 'RMSE'])
def generate_sample_2():
    """生成另一个样本集"""
    return [random.uniform(0, 10) for _ in range(20)] + ['a', 'b']

# 示例3：空数据集测试
@stats_decorator(['SUM', 'AVG'])
def empty_sample():
    """返回空数据集"""
    return ['no', 'numbers', None]

# 示例4：只有一个元素的样本
@stats_decorator(['SUM', 'AVG', 'VAR', 'RMSE'])
def single_value_sample():
    """只有一个数值的样本集"""
    return [42]

# 测试函数
def main():
    print("示例1: 混合数据类型样本")
    result1 = generate_sample_1()
    print(f"样本: [1.2, 3.5, 2.8, 4.1, 5.6, 'text', None]")
    print(f"统计结果: {result1}")
    
    print("\n示例2: 随机数据样本（部分统计项）")
    result2 = generate_sample_2()
    print(f"统计结果: {result2}")
    
    print("\n示例3: 空数据集测试")
    result3 = empty_sample()
    print(f"统计结果: {result3}")
    
    print("\n示例4: 单个数值样本")
    result4 = single_value_sample()
    print(f"统计结果: {result4}")

if __name__ == "__main__":
    main()
