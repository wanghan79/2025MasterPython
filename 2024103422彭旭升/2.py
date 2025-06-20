import random
import string
from typing import Any, Dict, List, Tuple, Union


def generate_random_samples(**kwargs):
    """
    构造任意嵌套数据类型随机样本生成函数

    参数说明:
    - count: 生成样本的个数，默认为1
    - structure: 数据结构定义，使用字典描述

    数据类型支持:
    - 'int': {'type': 'int', 'min': 0, 'max': 100}
    - 'float': {'type': 'float', 'min': 0.0, 'max': 1.0}
    - 'str': {'type': 'str', 'length': 10, 'charset': 'letters'}
    - 'bool': {'type': 'bool'}
    - 'list': {'type': 'list', 'length': 5, 'element': {...}}
    - 'tuple': {'type': 'tuple', 'length': 3, 'element': {...}}
    - 'dict': {'type': 'dict', 'keys': [...], 'values': {...}}
    - 'set': {'type': 'set', 'length': 4, 'element': {...}}
    """

    # 获取基本参数
    count = kwargs.get('count', 1)
    structure = kwargs.get('structure', {'type': 'int', 'min': 1, 'max': 10})

    def generate_single_value(spec):
        """根据规格生成单个值"""
        if not isinstance(spec, dict) or 'type' not in spec:
            return spec  # 如果不是规格字典，直接返回

        data_type = spec['type']

        if data_type == 'int':
            min_val = spec.get('min', 0)
            max_val = spec.get('max', 100)
            return random.randint(min_val, max_val)

        elif data_type == 'float':
            min_val = spec.get('min', 0.0)
            max_val = spec.get('max', 1.0)
            return round(random.uniform(min_val, max_val), 2)

        elif data_type == 'str':
            length = spec.get('length', 5)
            charset = spec.get('charset', 'letters')

            if charset == 'letters':
                chars = string.ascii_letters
            elif charset == 'digits':
                chars = string.digits
            elif charset == 'alphanumeric':
                chars = string.ascii_letters + string.digits
            else:
                chars = charset  # 自定义字符集

            return ''.join(random.choice(chars) for _ in range(length))

        elif data_type == 'bool':
            return random.choice([True, False])

        elif data_type == 'list':
            length = spec.get('length', 3)
            element_spec = spec.get('element', {'type': 'int'})
            return [generate_single_value(element_spec) for _ in range(length)]

        elif data_type == 'tuple':
            length = spec.get('length', 3)
            element_spec = spec.get('element', {'type': 'int'})
            return tuple(generate_single_value(element_spec) for _ in range(length))

        elif data_type == 'dict':
            keys = spec.get('keys', ['key1', 'key2', 'key3'])
            value_spec = spec.get('values', {'type': 'str'})

            result = {}
            for key in keys:
                if isinstance(key, dict):
                    # 如果key也是生成规格
                    actual_key = generate_single_value(key)
                else:
                    actual_key = key
                result[actual_key] = generate_single_value(value_spec)
            return result

        elif data_type == 'set':
            length = spec.get('length', 3)
            element_spec = spec.get('element', {'type': 'int'})
            # 生成比需要数量更多的元素，以防重复导致集合长度不足
            elements = []
            attempts = 0
            while len(set(elements)) < length and attempts < length * 10:
                elements.append(generate_single_value(element_spec))
                attempts += 1
            return set(elements[:length])

        elif data_type == 'mixed':
            # 混合类型，从多个规格中随机选择
            options = spec.get('options', [{'type': 'int'}, {'type': 'str'}])
            chosen_spec = random.choice(options)
            return generate_single_value(chosen_spec)

        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

    # 生成指定数量的样本
    samples = []
    for i in range(count):
        sample = generate_single_value(structure)
        samples.append(sample)

    return samples if count > 1 else samples[0]


def demo_usage():
    """演示函数的各种用法"""
    print("=" * 60)
    print("嵌套数据类型随机样本生成器演示")
    print("=" * 60)

    # 示例1: 简单整数
    print("\n1. 生成简单整数样本:")
    result = generate_random_samples(
        count=5,
        structure={'type': 'int', 'min': 1, 'max': 100}
    )
    print(f"结果: {result}")

    # 示例2: 字符串列表
    print("\n2. 生成字符串列表:")
    result = generate_random_samples(
        count=3,
        structure={
            'type': 'list',
            'length': 4,
            'element': {'type': 'str', 'length': 6, 'charset': 'alphanumeric'}
        }
    )
    print(f"结果: {result}")

    # 示例3: 复杂嵌套字典
    print("\n3. 生成复杂嵌套字典:")
    result = generate_random_samples(
        count=2,
        structure={
            'type': 'dict',
            'keys': ['name', 'age', 'scores', 'info'],
            'values': {
                'type': 'mixed',
                'options': [
                    {'type': 'str', 'length': 8},
                    {'type': 'int', 'min': 18, 'max': 65},
                    {
                        'type': 'list',
                        'length': 3,
                        'element': {'type': 'float', 'min': 60.0, 'max': 100.0}
                    },
                    {
                        'type': 'dict',
                        'keys': ['city', 'phone'],
                        'values': {'type': 'str', 'length': 10}
                    }
                ]
            }
        }
    )
    print(f"结果: {result}")

    # 示例4: 元组嵌套结构
    print("\n4. 生成元组嵌套结构:")
    result = generate_random_samples(
        count=2,
        structure={
            'type': 'tuple',
            'length': 3,
            'element': {
                'type': 'mixed',
                'options': [
                    {'type': 'int', 'min': 1, 'max': 10},
                    {
                        'type': 'list',
                        'length': 2,
                        'element': {'type': 'bool'}
                    },
                    {'type': 'str', 'length': 5}
                ]
            }
        }
    )
    print(f"结果: {result}")

    # 示例5: 学生信息样本(实际应用场景)
    print("\n5. 生成学生信息样本:")
    student_structure = {
        'type': 'dict',
        'keys': ['student_id', 'name', 'age', 'grades', 'contact'],
        'values': {
            'type': 'mixed',
            'options': [
                {'type': 'int', 'min': 20210001, 'max': 20219999},  # 学号
                {'type': 'str', 'length': 4, 'charset': 'letters'},  # 姓名
                {'type': 'int', 'min': 18, 'max': 25},  # 年龄
                {  # 成绩列表
                    'type': 'list',
                    'length': 5,
                    'element': {'type': 'float', 'min': 60.0, 'max': 100.0}
                },
                {  # 联系方式
                    'type': 'dict',
                    'keys': ['email', 'phone'],
                    'values': {'type': 'str', 'length': 11, 'charset': 'alphanumeric'}
                }
            ]
        }
    }

    students = generate_random_samples(count=3, structure=student_structure)
    print("学生信息样本:")
    for i, student in enumerate(students, 1):
        print(f"  学生{i}: {student}")

    # 示例6: 集合和多层嵌套
    print("\n6. 生成集合和多层嵌套:")
    result = generate_random_samples(
        count=1,
        structure={
            'type': 'dict',
            'keys': ['numbers', 'matrix', 'metadata'],
            'values': {
                'type': 'mixed',
                'options': [
                    {  # 数字集合
                        'type': 'set',
                        'length': 5,
                        'element': {'type': 'int', 'min': 1, 'max': 20}
                    },
                    {  # 二维矩阵
                        'type': 'list',
                        'length': 3,
                        'element': {
                            'type': 'list',
                            'length': 3,
                            'element': {'type': 'float', 'min': -1.0, 'max': 1.0}
                        }
                    },
                    {  # 元数据
                        'type': 'tuple',
                        'length': 2,
                        'element': {
                            'type': 'mixed',
                            'options': [
                                {'type': 'str', 'length': 8},
                                {'type': 'int', 'min': 100, 'max': 999}
                            ]
                        }
                    }
                ]
            }
        }
    )
    print(f"结果: {result}")


if __name__ == "__main__":
    demo_usage()