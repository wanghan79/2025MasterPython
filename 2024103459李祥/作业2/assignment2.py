# 作业2：Python函数封装
# 要求：构造任意嵌套数据类型随机样本生成函数
# 样本的嵌套数据结构及样本个数均由kwargs参数给入，函数返回相应的样本集

import random
import string
from typing import Any, Dict, List, Union

def generate_nested_samples(**kwargs) -> List[Any]:
    """
    生成任意嵌套数据类型的随机样本
    
    参数说明 (kwargs):
        sample_count (int): 样本个数，默认为10
        structure (dict): 数据结构定义，支持以下类型：
            - 'int': 整数，可指定范围 {'type': 'int', 'min': 1, 'max': 100}
            - 'float': 浮点数，可指定范围 {'type': 'float', 'min': 0.0, 'max': 1.0}
            - 'str': 字符串，可指定长度 {'type': 'str', 'length': 10}
            - 'bool': 布尔值 {'type': 'bool'}
            - 'list': 列表，可指定元素类型和长度 {'type': 'list', 'element_type': {...}, 'length': 5}
            - 'dict': 字典，可指定键值对结构 {'type': 'dict', 'keys': {...}}
            - 'tuple': 元组，可指定元素类型 {'type': 'tuple', 'elements': [...]}
    
    返回:
        List[Any]: 生成的样本列表
    
    示例:
        # 生成简单整数样本
        samples = generate_nested_samples(
            sample_count=5,
            structure={'type': 'int', 'min': 1, 'max': 100}
        )
        
        # 生成复杂嵌套结构样本
        samples = generate_nested_samples(
            sample_count=3,
            structure={
                'type': 'dict',
                'keys': {
                    'name': {'type': 'str', 'length': 8},
                    'age': {'type': 'int', 'min': 18, 'max': 65},
                    'scores': {
                        'type': 'list',
                        'element_type': {'type': 'float', 'min': 0.0, 'max': 100.0},
                        'length': 3
                    }
                }
            }
        )
    """
    
    # 获取参数
    sample_count = kwargs.get('sample_count', 10)
    structure = kwargs.get('structure', {'type': 'int', 'min': 1, 'max': 100})
    
    print(f"开始生成 {sample_count} 个样本...")
    print(f"数据结构: {structure}")
    
    samples = []
    for i in range(sample_count):
        sample = _generate_single_sample(structure)
        samples.append(sample)
        
        if (i + 1) % max(1, sample_count // 10) == 0:
            print(f"生成进度: {i + 1}/{sample_count}")
    
    print("样本生成完成!")
    return samples

def _generate_single_sample(structure: Dict[str, Any]) -> Any:
    """
    根据结构定义生成单个样本
    
    参数:
        structure: 数据结构定义字典
    
    返回:
        Any: 生成的样本数据
    """
    data_type = structure.get('type', 'int')
    
    if data_type == 'int':
        min_val = structure.get('min', 1)
        max_val = structure.get('max', 100)
        return random.randint(min_val, max_val)
    
    elif data_type == 'float':
        min_val = structure.get('min', 0.0)
        max_val = structure.get('max', 1.0)
        return round(random.uniform(min_val, max_val), 4)
    
    elif data_type == 'str':
        length = structure.get('length', 10)
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    
    elif data_type == 'bool':
        return random.choice([True, False])
    
    elif data_type == 'list':
        element_type = structure.get('element_type', {'type': 'int'})
        length = structure.get('length', 5)
        return [_generate_single_sample(element_type) for _ in range(length)]
    
    elif data_type == 'dict':
        keys_structure = structure.get('keys', {})
        result = {}
        for key, value_structure in keys_structure.items():
            result[key] = _generate_single_sample(value_structure)
        return result
    
    elif data_type == 'tuple':
        elements_structure = structure.get('elements', [{'type': 'int'}])
        return tuple(_generate_single_sample(elem_struct) for elem_struct in elements_structure)
    
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")

def demo_simple_samples():
    """演示简单数据类型样本生成"""
    print("=" * 60)
    print("演示1: 简单数据类型样本生成")
    print("=" * 60)
    
    # 整数样本
    int_samples = generate_nested_samples(
        sample_count=5,
        structure={'type': 'int', 'min': 1, 'max': 100}
    )
    print(f"整数样本: {int_samples}")
    
    # 浮点数样本
    float_samples = generate_nested_samples(
        sample_count=5,
        structure={'type': 'float', 'min': 0.0, 'max': 10.0}
    )
    print(f"浮点数样本: {float_samples}")
    
    # 字符串样本
    str_samples = generate_nested_samples(
        sample_count=3,
        structure={'type': 'str', 'length': 8}
    )
    print(f"字符串样本: {str_samples}")
    
    # 布尔值样本
    bool_samples = generate_nested_samples(
        sample_count=5,
        structure={'type': 'bool'}
    )
    print(f"布尔值样本: {bool_samples}")

def demo_nested_samples():
    """演示嵌套数据类型样本生成"""
    print("\n" + "=" * 60)
    print("演示2: 嵌套数据类型样本生成")
    print("=" * 60)
    
    # 列表样本
    list_samples = generate_nested_samples(
        sample_count=3,
        structure={
            'type': 'list',
            'element_type': {'type': 'int', 'min': 1, 'max': 10},
            'length': 4
        }
    )
    print(f"列表样本: {list_samples}")
    
    # 字典样本
    dict_samples = generate_nested_samples(
        sample_count=2,
        structure={
            'type': 'dict',
            'keys': {
                'name': {'type': 'str', 'length': 6},
                'age': {'type': 'int', 'min': 18, 'max': 65},
                'active': {'type': 'bool'}
            }
        }
    )
    print(f"字典样本: {dict_samples}")
    
    # 元组样本
    tuple_samples = generate_nested_samples(
        sample_count=3,
        structure={
            'type': 'tuple',
            'elements': [
                {'type': 'str', 'length': 4},
                {'type': 'int', 'min': 1, 'max': 100},
                {'type': 'float', 'min': 0.0, 'max': 1.0}
            ]
        }
    )
    print(f"元组样本: {tuple_samples}")

def demo_complex_nested_samples():
    """演示复杂嵌套数据类型样本生成"""
    print("\n" + "=" * 60)
    print("演示3: 复杂嵌套数据类型样本生成")
    print("=" * 60)
    
    # 复杂嵌套结构：学生信息
    complex_samples = generate_nested_samples(
        sample_count=2,
        structure={
            'type': 'dict',
            'keys': {
                'student_id': {'type': 'str', 'length': 8},
                'personal_info': {
                    'type': 'dict',
                    'keys': {
                        'name': {'type': 'str', 'length': 6},
                        'age': {'type': 'int', 'min': 18, 'max': 25},
                        'gender': {'type': 'str', 'length': 1}
                    }
                },
                'courses': {
                    'type': 'list',
                    'element_type': {
                        'type': 'dict',
                        'keys': {
                            'course_name': {'type': 'str', 'length': 10},
                            'score': {'type': 'float', 'min': 60.0, 'max': 100.0},
                            'credits': {'type': 'int', 'min': 1, 'max': 4}
                        }
                    },
                    'length': 3
                },
                'contact': {
                    'type': 'tuple',
                    'elements': [
                        {'type': 'str', 'length': 11},  # 电话号码
                        {'type': 'str', 'length': 20}   # 邮箱
                    ]
                }
            }
        }
    )
    
    print("复杂嵌套样本:")
    for i, sample in enumerate(complex_samples, 1):
        print(f"样本 {i}:")
        _pretty_print(sample, indent=2)
        print()

def _pretty_print(obj, indent=0):
    """美化打印嵌套数据结构"""
    spaces = " " * indent
    
    if isinstance(obj, dict):
        print(f"{spaces}{{")
        for key, value in obj.items():
            print(f"{spaces}  '{key}':", end=" ")
            if isinstance(value, (dict, list, tuple)):
                print()
                _pretty_print(value, indent + 4)
            else:
                print(f"{value}")
        print(f"{spaces}}}")
    
    elif isinstance(obj, list):
        print(f"{spaces}[")
        for item in obj:
            if isinstance(item, (dict, list, tuple)):
                _pretty_print(item, indent + 2)
            else:
                print(f"{spaces}  {item}")
        print(f"{spaces}]")
    
    elif isinstance(obj, tuple):
        print(f"{spaces}(")
        for item in obj:
            if isinstance(item, (dict, list, tuple)):
                _pretty_print(item, indent + 2)
            else:
                print(f"{spaces}  {item}")
        print(f"{spaces})")
    
    else:
        print(f"{spaces}{obj}")

def main():
    """主函数：运行所有演示"""
    print("Python嵌套数据类型随机样本生成函数演示")
    
    # 运行演示
    demo_simple_samples()
    demo_nested_samples()
    demo_complex_nested_samples()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
