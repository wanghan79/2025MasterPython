import random
import string
import datetime
from typing import Any, Dict, List, Tuple, Union, Callable
from functools import wraps
import math

# ======================= 作业2：数据生成函数 =======================
def generate_nested_samples(**kwargs) -> List[Dict[str, Any]]:
    # 验证样本数量
    if 'n' not in kwargs:
        raise ValueError("必须指定样本数量参数 'n'")

    n = kwargs.pop('n')
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n必须是正整数")
    # 数据结构定义
    structure = kwargs
    # 基本类型生成器
    def generate_basic(dtype: type) -> Any:
        if dtype == int:
            return random.randint(-1000, 1000)
        elif dtype == float:
            return round(random.uniform(-100.0, 100.0), 2)
        elif dtype == str:
            length = random.randint(5, 15)
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        elif dtype == bool:
            return random.choice([True, False])
        elif dtype == datetime.date:
            start = datetime.date(2000, 1, 1)
            end = datetime.date(2023, 12, 31)
            return start + datetime.timedelta(days=random.randint(0, (end - start).days))
        else:
            raise ValueError(f"不支持的数据类型: {dtype}")
    # 递归生成数据结构
    def generate_data(spec: Union[type, dict, list]) -> Any:
        # 处理简单类型
        if isinstance(spec, type):
            return generate_basic(spec)
        # 处理列表类型
        if isinstance(spec, dict) and spec.get('type') == list:
            element_spec = spec.get('element')
            if element_spec is None:
                raise ValueError("列表类型必须指定 'element' 参数")
            size = spec.get('size', random.randint(1, 5))
            return [generate_data(element_spec) for _ in range(size)]
        # 处理元组类型
        if isinstance(spec, dict) and spec.get('type') == tuple:
            elements_spec = spec.get('elements')
            if elements_spec is None:
                raise ValueError("元组类型必须指定 'elements' 参数")
            return tuple(generate_data(item) for item in elements_spec)
        # 处理字典类型
        if isinstance(spec, dict) and spec.get('type') == dict:
            fields_spec = spec.get('fields')
            if fields_spec is None:
                raise ValueError("字典类型必须指定 'fields' 参数")
            return {key: generate_data(value_spec) for key, value_spec in fields_spec.items()}
        # 处理混合结构列表
        if isinstance(spec, list):
            return [generate_data(item) for item in spec]
        # 处理固定值
        if isinstance(spec, dict) and spec.get('type') == 'fixed':
            return spec.get('value')
        # 处理直接传递的嵌套结构
        if isinstance(spec, dict):
            return {key: generate_data(value_spec) for key, value_spec in spec.items()}
        raise ValueError(f"无法解析的结构描述: {spec}")
    # 生成样本集
    samples = []
    for _ in range(n):
        sample = {}
        for field, field_spec in structure.items():
            sample[field] = generate_data(field_spec)
        samples.append(sample)
    return samples
  
# ======================= 作业3：统计装饰器 =======================
def stats_decorator(stats: List[str] = ['SUM', 'AVG', 'VAR', 'RMSE']):
    # 验证统计项
    valid_stats = {'SUM', 'AVG', 'VAR', 'RMSE'}
    for s in stats:
        if s not in valid_stats:
            raise ValueError(f"无效的统计项: {s}，可选: {valid_stats}")

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原始函数获取样本数据
            samples = func(*args, **kwargs)
            # 递归提取所有数值型数据
            all_numbers = []
            def extract_numbers(data):
                """递归提取数值型数据"""
                if isinstance(data, (int, float)):
                    all_numbers.append(data)
                elif isinstance(data, dict):
                    for value in data.values():
                        extract_numbers(value)
                elif isinstance(data, (list, tuple)):
                    for item in data:
                        extract_numbers(item)
            # 遍历所有样本提取数值
            for sample in samples:
                extract_numbers(sample)
            # 计算结果字典
            results = {}
            n = len(all_numbers)
            # 如果没有数值数据，返回None
            if n == 0:
                for stat in stats:
                    results[stat] = None
                return {'samples': samples, 'stats': results}
            # 计算基本统计量
            total = sum(all_numbers)
            mean = total / n
            # 计算请求的统计项
            if 'SUM' in stats:
                results['SUM'] = total
            if 'AVG' in stats:
                results['AVG'] = mean
            if 'VAR' in stats or 'RMSE' in stats:
                # 计算方差: 平方差的平均值
                variance = sum((x - mean) ** 2 for x in all_numbers) / n
                if 'VAR' in stats:
                    results['VAR'] = variance
                if 'RMSE' in stats:
                    # RMSE: 均方根误差 (方差的平方根)
                    results['RMSE'] = math.sqrt(variance)
            return {'samples': samples, 'stats': results}
        return wrapper
    return decorator
# ======================= 使用示例 =======================
if __name__ == "__main__":
    # 1. 定义复杂嵌套结构
    complex_structure = {
        'n': 5,  # 样本数量
        'id': int,
        'personal_info': {
            'name': str,
            'birthdate': datetime.date,
            'is_active': bool,
            'height': float,
            'weight': float
        },
        'scores': {
            'type': list,
            'element': {
                'subject': str,
                'score': float,
                'passed': bool
            },
            'size': 3
        },
        'metadata': {
            'type': tuple,
            'elements': [
                int,
                {'type': list, 'element': float, 'size': 2},
                {'type': dict, 'fields': {'key': str, 'value': bool}}
            ]
        },
        'tags': [str, str, str],  # 固定长度的字符串列表
        'rating': float
    }
    # 2. 应用装饰器（只计算平均值和均方根差）
    decorated_generator = stats_decorator(stats=['AVG', 'RMSE'])(generate_nested_samples)
    # 3. 生成样本并获取统计结果
    result = decorated_generator(**complex_structure)
    # 4. 打印样本和统计结果
    print("=" * 50 + "\n生成的样本:")
    for i, sample in enumerate(result['samples']):
        print(f"\n样本 {i + 1}:")
        for key, value in sample.items():
            print(f"  {key}: {value}")
    print("\n" + "=" * 50 + "\n统计结果:")
    for stat, value in result['stats'].items():
        print(f"{stat}: {value}")
    # 5. 使用完整统计（所有四项）
    print("\n" + "=" * 50 + "\n完整统计测试:")
    full_stats_generator = stats_decorator()(generate_nested_samples)
    full_result = full_stats_generator(**complex_structure)
    for stat, value in full_result['stats'].items():
        print(f"{stat}: {value}")
