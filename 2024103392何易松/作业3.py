import random
import string
import math
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, Union, List, Tuple

# -------- 优化1: 预计算常用字符串集合 --------
CHARACTERS = string.ascii_letters + string.digits  # 预计算一次

def random_string(length=8):
    """使用预计算的字符集生成随机字符串"""
    return ''.join(random.choices(CHARACTERS, k=length))

def random_date(start_year=2000, end_year=2025):
    """优化日期计算，避免重复创建日期对象"""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')

# -------- 优化2: 使用类型处理器映射替代if-else链 --------
TYPE_HANDLERS = {
    'int': lambda: random.randint(0, 10000),
    'float': lambda: round(random.uniform(0, 10000), 2),
    'str': random_string,
    'bool': lambda: random.choice([True, False]),
    'date': random_date
}

def generate_value(dtype: Union[str, list, tuple, dict]) -> Any:
    """优化后的递归生成函数"""
    if isinstance(dtype, str):
        # 使用类型处理器映射
        handler = TYPE_HANDLERS.get(dtype)
        if handler:
            return handler()
        raise ValueError(f"Unsupported type: {dtype}")

    elif isinstance(dtype, list):
        if not dtype:
            return []
        element_type = dtype[0]
        return [generate_value(element_type) for _ in range(random.randint(1, 3))]

    elif isinstance(dtype, tuple):
        return tuple(generate_value(t) for t in dtype)

    elif isinstance(dtype, dict):
        return {k: generate_value(v) for k, v in dtype.items()}

    else:
        raise TypeError(f"Invalid type: {type(dtype)}")

# -------- 数据生成主函数 --------
def DataSampler(sample_structure: dict, count: int = 1) -> list:
    """简化函数签名，移除不必要的kwargs"""
    return [generate_value(sample_structure) for _ in range(count)]

# -------- 优化3: 改进统计装饰器 --------
def stats_decorator(*stats_to_compute):
    """优化数值提取和统计计算"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            
            # 优化数值提取算法
            def extract_numeric_values(obj, values=None):
                """使用就地修改避免创建中间列表"""
                if values is None:
                    values = []
                
                if isinstance(obj, dict):
                    for v in obj.values():
                        extract_numeric_values(v, values)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        extract_numeric_values(item, values)
                elif isinstance(obj, (int, float)):
                    values.append(obj)
                
                return values
            
            all_values = []
            for sample in samples:
                extract_numeric_values(sample, all_values)
            
            result = {}
            n = len(all_values)
            
            if not all_values:
                return {"error": "No numeric values found."}
            
            # 优化统计计算（减少遍历次数）
            sum_val = sum(all_values)
            sum_sq = sum(x*x for x in all_values)
            
            if 'SUM' in stats_to_compute:
                result['SUM'] = sum_val
            
            if 'AVG' in stats_to_compute:
                result['AVG'] = sum_val / n
            
            if 'VAR' in stats_to_compute or 'RMSE' in stats_to_compute:
                mean = sum_val / n
                variance = sum_sq / n - mean*mean
                
                if 'VAR' in stats_to_compute:
                    result['VAR'] = variance
                
                if 'RMSE' in stats_to_compute:
                    result['RMSE'] = math.sqrt(sum_sq / n)
            
            return {
                'samples': samples,
                'statistics': result
            }
        
        return wrapper
    return decorator

# -------- 生成数据并自动分析数值字段 --------
@stats_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def generate_user_data(count=5):
    """使用更直观的参数传递"""
    structure = {
        'user_id': 'int',
        'name': 'str',
        'is_active': 'bool',
        'signup_date': 'date',
        'score': 'float',
        'tags': ['str'],
        'history': [
            {'item_id': 'int', 'timestamp': 'date', 'value': 'float'}
        ],
        'location': ('float', 'float')
    }
    return DataSampler(structure, count)

# -------- 优化4: 改进主程序输出 --------
if __name__ == '__main__':
    print("正在生成并分析用户数据...")
    result = generate_user_data(count=5)
    
    print("\n生成的数据样本:")
    for i, sample in enumerate(result['samples'], 1):
        print(f"样本 {i}:")
        for key, value in sample.items():
            print(f"  {key}: {value}")
        print()
    
    print("\n数值统计结果:")
    for stat, value in result['statistics'].items():
        # 根据统计类型格式化输出
        if stat in {'AVG', 'VAR', 'RMSE'}:
            print(f"{stat}: {value:.4f}")
        else:
            print(f"{stat}: {value}")
    
    # 显示提取的数值总数
    print(f"\n提取的数值总数: {len([v for s in result['samples'] for v in extract_numeric_values(s)])}")
