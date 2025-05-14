import random
import string

def sampling(**kwargs):
    """递归解析字典中的 'node'、'range' 和 'subs'，进行采样"""
    node = kwargs.get("node")  # 当前节点类型
    range_values = kwargs.get("range", None)  # 获取范围值
    subs = kwargs.get("subs", [])  # 获取子节点列表

    samples = []  # 存储采样结果

    # 处理当前层数据
    if node:
        if isinstance(range_values, (tuple, list)) and len(range_values) == 2:
            # 数值类型 (min, max)
            if isinstance(range_values[0], int) and isinstance(range_values[1], int):
                samples.append(random.randint(*range_values))
            elif isinstance(range_values[0], float) and isinstance(range_values[1], float):
                samples.append(random.uniform(*range_values))
        elif isinstance(range_values, str):
            # 字符串类型
            length = kwargs.get("length", 10)  # 默认长度
            samples.append(''.join(random.choices(range_values, k=length)))

    # 递归解析子节点
    for sub in subs:
        samples.extend(sampling(**sub))

    return samples

# **示例数据树**
data_tree = {
    "node": "root",
    "subs": [
        {"node": "int", "range": (1, 100)},
        {"node": "float", "range": (0.1, 99.9)},
        {"node": "str", "range": string.ascii_letters, "length": 8},
        {"node": "nested", "subs": [
            {"node": "int", "range": (200, 500)},
            {"node": "str", "range": string.digits, "length": 6}
        ]}
    ]
}

# **执行采样**
sample_data = sampling(**data_tree)
print(sample_data)