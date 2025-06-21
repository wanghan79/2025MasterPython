import random
from typing import Any, Dict, List, Union

def generate_samples(**kwargs) -> List[Any]:
    """
    生成任意嵌套结构的样本集
    参数:
        num_samples: 样本数量
        max_depth: 最大嵌套深度
        types: 允许的基础类型列表
    """
    # 获得参数
    num_samples = kwargs.get('num_samples')
    max_depth = kwargs.get('max_depth')
    allowed_types = kwargs.get('types')
    # 生成
    samples = []
    for _ in range(num_samples):
        samples.append(generate_nested_structure(max_depth, allowed_types))
    return samples

def generate_nested_structure(current_depth: int, allowed_types: List[type]) -> Any:
    """递归生成嵌套结构的辅助函数"""
    # 终止条件：达到最大深度时生成基础类型
    if current_depth <= 0:
        if int in allowed_types:
            return random.randint(0, 100)
        elif float in allowed_types:
            return round(random.uniform(0, 100), 2)
        elif str in allowed_types:
            return ''.join(random.choices('abcdefg', k=random.randint(3, 6)))
        else:
            return random.randint(0, 100)  # 默认返回整数

    # 递归生成容器类型
    container_type = random.choice(['list', 'dict', 'tuple'])
    if container_type == 'list':
        length = random.randint(1, 5)
        return [generate_nested_structure(current_depth - 1, allowed_types) for _ in range(length)]
    elif container_type == 'dict':
        length = random.randint(1, 5)
        return {f'key_{i}': generate_nested_structure(current_depth - 1, allowed_types) for i in range(length)}
    elif container_type == 'tuple':
        length = random.randint(1, 5)
        return tuple(generate_nested_structure(current_depth - 1, allowed_types) for _ in range(length))

if __name__ == '__main__':
    samples = generate_samples(num_samples=5,max_depth=2,types=[int,float])
    print(samples)