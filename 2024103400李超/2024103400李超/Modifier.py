import random
import string
import math

# ----------------- 三层装饰器 -----------------
def statistics(*stats):
    """带参数的三层装饰器，用于统计采样数据"""
   
    def decorator(func):
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)  # 调用被修饰函数，拿到采样数据
            results = {}  # 用字典存各项统计
           
            # 扁平化数据，只保留数值型（int/float）
            numeric_data = [x for x in data if isinstance(x, (int, float))]
            print(numeric_data)
           
            if not numeric_data:
                return {stat: None for stat in stats}  # 没有数值，返回None
           
            if 'sum' in stats:
                results['sum'] = sum(numeric_data)
            if 'mean' in stats:
                results['mean'] = sum(numeric_data) / len(numeric_data)
            if 'variance' in stats:
                mean_val = sum(numeric_data) / len(numeric_data)
                results['variance'] = sum((x - mean_val) ** 2 for x in numeric_data) / len(numeric_data)
            if 'rmse' in stats:
                mean_val = sum(numeric_data) / len(numeric_data)
                mse = sum((x - mean_val) ** 2 for x in numeric_data) / len(numeric_data)
                results['rmse'] = math.sqrt(mse)
               
            return results
       
        return wrapper
   
    return decorator

# ----------------- 数据采样函数 -----------------
def sampling(**kwargs):
    """递归解析字典中的 'node'、'range' 和 'subs'，进行采样"""
    node = kwargs.get("node")
    range_values = kwargs.get("range", None)
    subs = kwargs.get("subs", [])

    samples = []

    # 处理当前节点
    if node:
        if isinstance(range_values, (tuple, list)) and len(range_values) == 2:
            if isinstance(range_values[0], int) and isinstance(range_values[1], int):
                samples.append(random.randint(*range_values))
            elif isinstance(range_values[0], float) and isinstance(range_values[1], float):
                samples.append(random.uniform(*range_values))
        elif isinstance(range_values, str):
            length = kwargs.get("length", 10)
            samples.append(''.join(random.choices(range_values, k=length)))

    # 递归子节点
    for sub in subs:
        samples.extend(sampling(**sub))

    return samples

# ----------------- 示例 -----------------
# 示例数据树
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

# 使用装饰器：指定要统计的内容
@statistics('sum', 'mean', 'variance', 'rmse')
def sampled_data():
    sample_data = sampling(**data_tree)
    print(sample_data)
    # return sampling(**data_tree)
    return sample_data

# 执行
results = sampled_data()
print(results)