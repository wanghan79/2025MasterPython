import random
import string
import collections
from typing import Any, Dict, List, Union


def generate_nested_samples(n_samples: int, **kwargs) -> List[Any]:
    """
    生成任意嵌套结构的随机样本集

    参数:
        n_samples: 需要生成的样本数量
        **kwargs: 描述样本结构的参数，支持多层嵌套

    返回:
        包含n_samples个样本的列表，每个样本符合指定的嵌套结构

    示例:
        # 生成5个包含字典和列表嵌套的样本
        samples = generate_nested_samples(
            5,
            name={'type': 'str', 'length': 8},
            age={'type': 'int', 'range': [18, 65]},
            scores={'type': 'list', 'length': 3, 'element': {'type': 'float', 'range': [0.0, 10.0]}},
            address={
                'type': 'dict',
                'fields': {
                    'city': {'type': 'choice', 'options': ['北京', '上海', '广州', '深圳']},
                    'zipcode': {'type': 'str', 'length': 6, 'chars': '0123456789'}
                }
            }
        )
    """
    # 输入验证
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    if not kwargs:
        raise ValueError("At least one field must be specified in kwargs")

    return [_generate_sample(kwargs) for _ in range(n_samples)]


def _generate_sample(spec: Dict[str, Any]) -> Any:
    """根据规范生成单个样本"""
    # 处理不同类型的数据生成
    data_type = spec.get('type', None)

    if data_type == 'int':
        return _generate_int(spec)
    elif data_type == 'float':
        return _generate_float(spec)
    elif data_type == 'str':
        return _generate_string(spec)
    elif data_type == 'bool':
        return random.choice([True, False])
    elif data_type == 'choice':
        return random.choice(spec['options'])
    elif data_type == 'list':
        return _generate_list(spec)
    elif data_type == 'dict':
        return _generate_dict(spec)
    elif data_type is None and isinstance(spec, dict) and 'fields' not in spec:
        # 处理嵌套字典规范
        return {key: _generate_sample(value) for key, value in spec.items()}
    else:
        raise ValueError(f"Unsupported data type specification: {spec}")


def _generate_int(spec: Dict[str, Any]) -> int:
    """生成随机整数"""
    if 'range' in spec:
        return random.randint(spec['range'][0], spec['range'][1])
    elif 'values' in spec:
        return random.choice(spec['values'])
    else:
        return random.randint(0, 100)  # 默认范围


def _generate_float(spec: Dict[str, Any]) -> float:
    """生成随机浮点数"""
    if 'range' in spec:
        return random.uniform(spec['range'][0], spec['range'][1])
    elif 'values' in spec:
        return random.choice(spec['values'])
    else:
        return random.random()  # 默认0-1之间


def _generate_string(spec: Dict[str, Any]) -> str:
    """生成随机字符串"""
    length = spec.get('length', 10)
    charset = spec.get('chars', string.ascii_letters + string.digits)
    return ''.join(random.choices(charset, k=length))


def _generate_list(spec: Dict[str, Any]) -> List[Any]:
    """生成随机列表"""
    length = spec.get('length', random.randint(1, 5))  # 默认长度1-5
    element_spec = spec.get('element', {'type': 'int'})  # 默认整数元素

    return [_generate_sample(element_spec) for _ in range(length)]


def _generate_dict(spec: Dict[str, Any]) -> Dict[str, Any]:
    """生成随机字典"""
    fields = spec.get('fields', {})
    return {key: _generate_sample(value) for key, value in fields.items()}


if __name__ == "__main__":
    # 示例用法
    samples = generate_nested_samples(
        3,  # 生成3个样本
        id={'type': 'int', 'range': [1000, 9999]},
        name={'type': 'str', 'length': 8},
        active={'type': 'bool'},
        attributes={
            'type': 'dict',
            'fields': {
                'age': {'type': 'int', 'range': [18, 65]},
                'weight': {'type': 'float', 'range': [50.0, 100.0]},
                'tags': {'type': 'list', 'length': 3, 'element': {'type': 'str', 'length': 4}}
            }
        },
        history={
            'type': 'list',
            'length': 2,
            'element': {
                'date': {'type': 'str', 'chars': '0123456789', 'length': 8},
                'event': {'type': 'choice', 'options': ['login', 'purchase', 'logout']}
            }
        }
    )

    # 打印生成的样本
    for i, sample in enumerate(samples):
        print(f"Sample {i + 1}:")
        print(sample)
        print("-" * 50)