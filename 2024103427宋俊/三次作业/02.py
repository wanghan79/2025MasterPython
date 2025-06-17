import random
import string
import datetime
from typing import Any, Dict, List, Tuple, Union

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

# 示例用法
if __name__ == "__main__":
    # 定义复杂嵌套结构
    complex_structure = {
        'n': 5,  # 样本数量
        'id': int,
        'personal_info': {
            'name': str,
            'birthdate': datetime.date,
            'is_active': bool
        },
        'scores': {
            'type': list,
            'element': {
                'subject': str,
                'score': float,
                'passed': bool
            },
            'size': 2
        },
        'metadata': {
            'type': tuple,
            'elements': [
                int,
                {'type': list, 'element': float},
                {'type': dict, 'fields': {'key': str, 'value': bool}}
            ]
        },
        'tags': [str, str, str]  # 固定长度的字符串列表
    }

    # 生成样本
    samples = generate_nested_samples(**complex_structure)

    # 打印结果
    for i, sample in enumerate(samples):
        print(f"\n样本 {i + 1}:")
        for key, value in sample.items():
            print(f"  {key}: {value}")
        print("-" * 50)
