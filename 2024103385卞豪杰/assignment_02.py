# coding=utf-8
import random
import string
from datetime import datetime, timedelta
from typing import Any, Union

def generate_samples(**kwargs) -> list:
    """
    生成任意嵌套结构的随机样本数据集

    参数:
        size: 生成的样本数量
        schema: 描述数据结构的嵌套定义

    返回:
        包含指定数量样本的列表，每个样本符合给定的数据结构

    示例:
        # 生成10个样本，每个样本是包含字符串和嵌套字典的元组
        samples = generate_samples(
            size=10,
            schema=(
                str,
                {
                    'id': int,
                    'scores': [float],
                    'details': {'active': bool, 'created': datetime}
                }
            )
        )
    """
    def generate_value(data_type: Union[type, list, tuple, dict]) -> Any:
        """根据类型描述生成随机值"""
        # 处理基本数据类型
        if data_type is int:
            return random.randint(1, 1000)

        elif data_type is float:
            return round(random.uniform(0.0, 100.0), 2)

        elif data_type is str:
            length = random.randint(5, 15)
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

        elif data_type is bool:
            return random.choice([True, False])

        elif data_type is datetime:
            start = datetime(2000, 1, 1)
            end = datetime(2023, 12, 31)
            return start + timedelta(days=random.randint(0, (end - start).days))

        # 处理列表类型
        elif isinstance(data_type, list):
            # 列表长度随机 (1-5个元素)
            return [generate_value(data_type[0]) for _ in range(random.randint(1, 5))]

        # 处理元组类型
        elif isinstance(data_type, tuple):
            return tuple(generate_value(item) for item in data_type)

        # 处理字典类型
        elif isinstance(data_type, dict):
            return {key: generate_value(value) for key, value in data_type.items()}

        # 处理嵌套结构
        elif callable(data_type):  # 处理自定义嵌套函数
            return data_type()

        else:
            raise TypeError(f"不支持的数类型: {type(data_type)}")

    # 验证必需参数
    if 'size' not in kwargs or 'schema' not in kwargs:
        raise ValueError("必须包含'size'和'schema'参数")

    size = kwargs['size']
    schema = kwargs['schema']

    # 生成样本数据集
    return [generate_value(schema) for _ in range(size)]


# 示例使用
if __name__ == "__main__":
    # 自定义嵌套生成器函数
    def nested_structure():
        return {
            'matrix': [[random.randint(1, 100) for _ in range(3)] for _ in range(3)],
            'metadata': (
                ''.join(random.choices(string.ascii_uppercase, k=4)),
                random.choice([True, False])
            )
        }

    # 定义复杂的数据结构
    complex_schema = {
        'user_id': int,
        'username': str,
        'profile': {
            'email': str,
            'age': int,
            'premium': bool,
            'joined': datetime
        },
        'transactions': [{
            'id': str,
            'amount': float,
            'items': [str],
            'completed': bool
        }],
        'nested_data': nested_structure  # 使用自定义嵌套生成器
    }

    # 生成5个样本
    samples = generate_samples(
        size=5,
        schema=complex_schema
    )

    # 打印第一个样本
    import pprint
    print("生成的样本示例:")
    pprint.pprint(samples[0], width=120, sort_dicts=False)