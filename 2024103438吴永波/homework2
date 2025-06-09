import random
import string
from typing import Any, Callable

def random_int():
    return random.randint(0, 100)

def random_float():
    return random.uniform(0, 100)

def random_str():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def random_bool():
    return random.choice([True, False])

def random_sample(structure: Any) -> Any:
    """
    递归生成与structure结构一致的随机数据。
    structure可以是类型、list/tuple/dict嵌套结构。
    """
    if isinstance(structure, type):
        if structure == int:
            return random_int()
        elif structure == float:
            return random_float()
        elif structure == str:
            return random_str()
        elif structure == bool:
            return random_bool()
        else:
            raise ValueError(f"不支持的类型: {structure}")
    elif isinstance(structure, list):
        # 结构如: [int, float, ...]，每个元素为类型或嵌套结构
        return [random_sample(item) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(random_sample(item) for item in structure)
    elif isinstance(structure, dict):
        # 结构如: {"a": int, "b": [float, str]}
        return {k: random_sample(v) for k, v in structure.items()}
    else:
        raise ValueError(f"不支持的结构: {structure}")

def random_nested_sample(num: int = 1, structure: Any = int, **kwargs) -> list:
    """
    生成指定嵌套结构的随机样本集。
    :param num: 样本个数
    :param structure: 嵌套结构描述（类型、list、tuple、dict等）
    :param kwargs: 也可用structure=...指定结构，num=...指定样本数
    :return: 样本集（list）
    """
    # 兼容通过kwargs传参
    if 'structure' in kwargs:
        structure = kwargs['structure']
    if 'num' in kwargs:
        num = kwargs['num']
    return [random_sample(structure) for _ in range(num)]

# 示例用法
if __name__ == "__main__":
    # 结构示例1：字典嵌套列表和元组
    struct1 = {
        "id": int,
        "name": str,
        "scores": [float, float, float],
        "tags": (str, bool),
        "meta": {"active": bool, "level": int}
    }
    print("样本结构1:")
    samples1 = random_nested_sample(num=3, structure=struct1)
    for s in samples1:
        print(s)
    print("\n")

    # 结构示例2：列表嵌套字典
    struct2 = [
        {"x": int, "y": float},
        {"flag": bool, "desc": str}
    ]
    print("样本结构2:")
    samples2 = random_nested_sample(num=2, structure=struct2)
    for s in samples2:
        print(s)
    print("\n")

    # 结构示例3：元组嵌套列表和字典
    struct3 = (
        [int, int, int],
        {"a": float, "b": [str, bool]}
    )
    print("样本结构3:")
    samples3 = random_nested_sample(num=2, structure=struct3)
    for s in samples3:
        print(s)
    print("\n")

    # 结构示例4：单一类型
    print("样本结构4:")
    samples4 = random_nested_sample(num=5, structure=str)
    for s in samples4:
        print(s)
    print("\n") 
