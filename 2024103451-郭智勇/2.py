import random
import string

def generate_random_samples(**kwargs):
    """
    生成随机嵌套数据的样本集合。

    参数:
    - kwargs: 包含以下几个参数:
        - 'structure': 指定嵌套数据结构的配置，例如 [{'type': 'list', 'length': 5}, {'type': 'dict', 'keys': 3, 'values': [{'type': 'int', 'min': 1, 'max': 100}]}]
        - 'count': 需要生成的样本数量
        - 'max_nesting_depth': 最大嵌套深度（可选，默认为3）
        - 'rng': 随机数生成器（可选，默认使用random）

    返回:
        list: 随机生成的嵌套数据样本集合
    """
    # 使用kwargs中的参数（如果未指定则使用默认值）
    structure = kwargs.get('structure', [{'type': 'list', 'length': 10}])
    count = kwargs.get('count', 1)
    max_depth = kwargs.get('max_nesting_depth', 3)
    rng = kwargs.get('rng', random)

    samples = []

    for _ in range(count):
        sample = generate_random_data(structure, max_depth, rng)
        samples.append(sample)

    return samples

def generate_random_data(structure, max_depth, rng, current_depth=0):
    """
    递归生成符合特定结构的随机数据。

    参数:
        structure: 数据结构配置
        max_depth: 最大嵌套深度
        rng: 随机数生成器
        current_depth: 当前嵌套深度（默认0）

    返回:
        生成的随机数据
    """
    data = []

    for layer in structure:
        if current_depth >= max_depth:
            # 如果达到最大嵌套深度，生成基本类型
            data.append(generate_type(layer, rng))
        else:
            data_type = layer.get('type', 'list')
            if data_type == 'list':
                length = layer.get('length', rng.randint(1, 10))
                nested_structure = layer.get('elements', [{'type': 'int'}])
                data.append([generate_random_data(nested_structure, max_depth, rng, current_depth + 1) 
                             for _ in range(length)])
            elif data_type == 'dict':
                keys = layer.get('keys', rng.randint(1, 5))
                key_type = layer.get('key_type', 'str')
                nested_structure = layer.get('values', [{'type': 'int'}])
                data_dict = {}
                for _ in range(keys):
                    key = generate_type({'type': key_type}, rng)
                    value = generate_random_data(nested_structure, max_depth, rng, current_depth + 1)
                    data_dict[key] = value
                data.append(data_dict)
            else:
                # 生成基本类型
                data.append(generate_type(layer, rng))

    return data[0] if len(data) == 1 else data

def generate_type(config, rng):
    """
    根据配置生成特定类型的随机值。

    参数:
        config: 类型配置
        rng: 随机数生成器
    """
    data_type = config.get('type', 'int')

    if data_type == 'int':
        return rng.randint(config.get('min', 0), config.get('max', 100))
    elif data_type == 'float':
        return rng.uniform(config.get('min', 0.0), config.get('max', 100.0))
    elif data_type == 'str':
        length = config.get('length', rng.randint(1, 10))
        letters = string.ascii_letters
        return ''.join(rng.choice(letters) for _ in range(length))
    elif data_type == 'bool':
        return rng.choice([True, False])
    elif data_type == 'none':
        return None
    elif data_type == 'custom':
        # 支持自定义生成函数
        return config.get('generator', lambda: None)()

    # 默认返回整数
    return rng.randint(0, 100)

# 示例使用：
if __name__ == "__main__":
    # 定义结构配置
    structure_config = [
        {'type': 'list', 'length': 3, 'elements': [
            {'type': 'dict', 'keys': 2, 'values': [
                {'type': 'int', 'min': 10, 'max': 100},
                {'type': 'str', 'length': 5}
            ]}
        ]},
        {'type': 'int'},
        {'type': 'float'}
    ]

    # 生成5个样本
    samples = generate_random_samples(structure=structure_config, count=5, max_nesting_depth=2)
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample}\n")