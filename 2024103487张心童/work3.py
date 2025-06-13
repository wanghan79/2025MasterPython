import math
import random
from functools import wraps
from typing import List, Dict, Union, Callable

# ==================== 作业二的数据生成部分 ====================
class DataGenerator:
    """作业二封装的随机数据生成器"""
    
    @staticmethod
    def generate_random_value(data_type: str):
        """生成单个随机值"""
        if data_type == 'int':
            return random.randint(1, 100)
        elif data_type == 'float':
            return round(random.uniform(0.1, 99.9), 2)
        elif data_type == 'str':
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
        elif data_type == 'bool':
            return random.choice([True, False])
        else:
            raise ValueError(f"Unsupported type: {data_type}")

    @staticmethod
    def generate_nested_data(schema: Dict) -> Dict:
        """生成嵌套结构数据"""
        def _generate(data_schema):
            if isinstance(data_schema, dict):
                return {k: _generate(v) for k, v in data_schema.items()}
            elif isinstance(data_schema, list):
                return [_generate(item) for item in data_schema]
            elif isinstance(data_schema, tuple):
                return tuple(_generate(item) for item in data_schema)
            elif isinstance(data_schema, str) and data_schema in ['int', 'float', 'str', 'bool']:
                return DataGenerator.generate_random_value(data_schema)
            else:
                return data_schema
        
        return _generate(schema)

    @classmethod
    def generate_samples(cls, schema: Dict, count: int = 1) -> List[Dict]:
        """生成指定数量的样本"""
        return [cls.generate_nested_data(schema) for _ in range(count)]


# ==================== 作业三的统计装饰器部分 ====================
class StatsAnalyzer:
    """统计分析工具类"""
    
    @staticmethod
    def analyze(data: Union[Dict, List], stats: List[str]) -> Dict[str, float]:
        """
        自动分析数据中的数值型叶节点，返回统计结果
        :param data: 输入数据（单个样本或样本列表）
        :param stats: 需要计算的统计项 ['SUM', 'AVG', 'VAR', 'RMSE']
        :return: 统计结果字典
        """
        # 收集所有数值型叶节点
        numeric_values = []
        
        def _collect_numbers(d):
            if isinstance(d, (int, float)):
                numeric_values.append(d)
            elif isinstance(d, dict):
                for v in d.values():
                    _collect_numbers(v)
            elif isinstance(d, (list, tuple)):
                for item in d:
                    _collect_numbers(item)
        
        _collect_numbers(data)
        
        if not numeric_values:
            return {}
        
        # 计算各项统计指标
        results = {}
        n = len(numeric_values)
        total = sum(numeric_values)
        
        if 'SUM' in stats:
            results['SUM'] = total
            
        if 'AVG' in stats:
            results['AVG'] = total / n
            
        if 'VAR' in stats or 'RMSE' in stats:
            mean = total / n
            variance = sum((x - mean) ** 2 for x in numeric_values) / n
            if 'VAR' in stats:
                results['VAR'] = variance
            if 'RMSE' in stats:
                results['RMSE'] = math.sqrt(variance)
        
        return results


def stats_decorator(*requested_stats: str):
    """
    带参数的统计装饰器
    :param requested_stats: 需要计算的统计项 ('SUM', 'AVG', 'VAR', 'RMSE')
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 先执行原始函数获取数据
            samples = func(*args, **kwargs)
            
            # 2. 对每个样本进行统计分析
            all_results = []
            for sample in samples:
                stats_result = StatsAnalyzer.analyze(sample, list(requested_stats))
                all_results.append({
                    'data': sample,
                    'stats': stats_result
                })
            
            # 3. 返回包含原始数据和统计结果的结构
            return all_results
        return wrapper
    return decorator


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 1. 定义数据生成schema（作业二部分）
    user_schema = {
        "user_id": "int",
        "user_info": {
            "name": "str",
            "age": "int",
            "balance": "float",
            "is_vip": "bool"
        },
        "transactions": [
            {"amount": "float", "valid": "bool"}
        ]
    }

    # 2. 应用统计装饰器（作业三部分）
    @stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
    def generate_user_data(count: int):
        """被装饰的原始数据生成函数"""
        return DataGenerator.generate_samples(user_schema, count)

    # 3. 生成带统计结果的数据
    results = generate_user_data(3)
    
    # 4. 打印结果
    for i, result in enumerate(results, 1):
        print(f"\n=== 样本 {i} ===")
        print("原始数据:", result['data'])
        print("统计结果:", result['stats'])