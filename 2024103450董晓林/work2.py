# coding=utf-8
import random
from typing import Dict, List, Union, Tuple, Any

def generate_random_value(value_range: Union[Tuple[int, int], Tuple[float, float]]) -> Union[int, float]:
    """根据范围生成随机值
    
    Args:
        value_range: 取值范围元组 (min, max)
    
    Returns:
        生成的随机值
    """
    if all(isinstance(x, int) for x in value_range):
        return random.randint(value_range[0], value_range[1])
    elif all(isinstance(x, float) for x in value_range):
        return random.uniform(value_range[0], value_range[1])
    return value_range

def process_nested_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """处理嵌套字典，生成随机值
    
    Args:
        data: 输入的数据字典
    
    Returns:
        处理后的字典
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = process_nested_dict(value)
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            result[key] = generate_random_value(value)
        else:
            result[key] = value
    return result

def generate_data(num: int, **kwargs) -> List[Dict[str, Any]]:
    """生成指定数量的随机数据
    
    Args:
        num: 需要生成的数据数量
        **kwargs: 数据格式定义
    
    Returns:
        生成的数据列表
    """
    result = []
    for i in range(num):
        element = {}
        for key, value in kwargs.items():
            processed_value = process_nested_dict(value) if isinstance(value, dict) else value
            element[f"{key}{i}"] = processed_value
        result.append(element)
    return result

def main():
    # 数据格式定义
    data_format = {
        'town': {
            'school': {
                'teachers': (50, 70),
                'students': (800, 1200),
                'others': (20, 40),
                'money': (410000.5, 986553.1)
            },
            'hospital': {
                'docters': (40, 60),
                'nurses': (60, 80),
                'patients': (200, 300),
                'money': (110050.5, 426553.4)
            },
            'supermarket': {
                'sailers': (80, 150),
                'shop': (30, 60),
                'money': (310000.3, 7965453.4)
            }
        }
    }
    
    # 生成数据
    num_data = 5
    result = generate_data(num_data, **data_format)
    
    # 打印结果
    print("\n=== 生成的数据 ===")
    for i, data in enumerate(result, 1):
        print(f"\n数据 {i}:")
        for key, value in data.items():
            print(f"{key}: {value}")

if __name__ == '__main__':
    main()
