import random
import string
from typing import Any, Dict, List, Union

def generate_random_value(data_type: str) -> Any:
    """生成随机值"""
    if data_type == 'int':
        return random.randint(-1000, 1000)
    elif data_type == 'float':
        return random.uniform(-1000, 1000)
    elif data_type == 'str':
        return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 10)))
    elif data_type == 'bool':
        return random.choice([True, False])
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")

def generate_nested_sample(structure: Dict[str, Any]) -> Any:
    """根据结构定义生成嵌套数据样本"""
    if 'type' not in structure:
        raise ValueError("结构定义必须包含 'type' 字段")
    
    data_type = structure['type']
    
    if data_type in ['int', 'float', 'str', 'bool']:
        return generate_random_value(data_type)
    
    elif data_type == 'list':
        if 'items' not in structure:
            raise ValueError("list 类型必须包含 'items' 字段")
        length = structure.get('length', random.randint(1, 5))
        return [generate_nested_sample(structure['items']) for _ in range(length)]
    
    elif data_type == 'dict':
        if 'properties' not in structure:
            raise ValueError("dict 类型必须包含 'properties' 字段")
        return {
            key: generate_nested_sample(value)
            for key, value in structure['properties'].items()
        }
    
    elif data_type == 'tuple':
        if 'items' not in structure:
            raise ValueError("tuple 类型必须包含 'items' 字段")
        length = structure.get('length', random.randint(1, 5))
        return tuple(generate_nested_sample(structure['items']) for _ in range(length))
    
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")

def generate_samples(**kwargs) -> List[Any]:
    """生成随机样本集
    
    Args:
        **kwargs: 包含以下参数：
            - structure: 数据结构定义
            - count: 样本数量（可选，默认为1）
    
    Returns:
        List[Any]: 生成的样本列表
    """
    if 'structure' not in kwargs:
        raise ValueError("必须提供 'structure' 参数")
    
    structure = kwargs['structure']
    count = kwargs.get('count', 1)
    
    return [generate_nested_sample(structure) for _ in range(count)]

# 使用示例
if __name__ == "__main__":
    # 示例1：生成包含嵌套列表和字典的样本
    structure1 = {
        "type": "dict",
        "properties": {
            "name": {"type": "str"},
            "age": {"type": "int"},
            "scores": {
                "type": "list",
                "items": {
                    "type": "dict",
                    "properties": {
                        "subject": {"type": "str"},
                        "score": {"type": "float"}
                    }
                }
            }
        }
    }
    
    # 生成3个样本
    samples1 = generate_samples(structure=structure1, count=3)
    print("示例1 - 嵌套字典和列表:")
    for i, sample in enumerate(samples1, 1):
        print(f"样本 {i}:", sample)
    
    # 示例2：生成包含元组的嵌套结构
    structure2 = {
        "type": "list",
        "items": {
            "type": "tuple",
            "items": {
                "type": "dict",
                "properties": {
                    "id": {"type": "int"},
                    "value": {"type": "float"}
                }
            }
        }
    }
    
    # 生成2个样本
    samples2 = generate_samples(structure=structure2, count=2)
    print("\n示例2 - 包含元组的嵌套结构:")
    for i, sample in enumerate(samples2, 1):
        print(f"样本 {i}:", sample)
