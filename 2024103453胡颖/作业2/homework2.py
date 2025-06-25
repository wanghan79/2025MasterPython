import random
import string

def generate_random_value(data_type, depth, max_depth, list_avg_len=3, dict_avg_len=3):

    if data_type == 'int':
        return random.randint(0, 100)
    elif data_type == 'float':
        return round(random.uniform(0.0, 100.0), 2)
    elif data_type == 'str':
        length = random.randint(5, 15)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    elif data_type == 'bool':
        return random.choice([True, False])
    elif data_type == 'list':
        if depth >= max_depth:
            return [generate_random_value(random.choice(['int', 'float', 'str', 'bool']), depth + 1, max_depth)
                    for _ in range(random.randint(1, list_avg_len * 2))]
        else:
            possible_types = ['int', 'float', 'str', 'bool', 'list', 'dict']
            return [generate_random_value(random.choice(possible_types), depth + 1, max_depth, list_avg_len, dict_avg_len)
                    for _ in range(random.randint(1, list_avg_len * 2))]
    elif data_type == 'dict':
        if depth >= max_depth:
            return {
                f"key_{i}": generate_random_value(random.choice(['int', 'float', 'str', 'bool']), depth + 1, max_depth)
                for i in range(random.randint(1, dict_avg_len * 2))
            }
        else:
            possible_types = ['int', 'float', 'str', 'bool', 'list', 'dict']
            return {
                f"key_{i}": generate_random_value(random.choice(possible_types), depth + 1, max_depth, list_avg_len, dict_avg_len)
                for i in range(random.randint(1, dict_avg_len * 2))
            }
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")

def generate_nested_sample_set(**kwargs):

    sample_count = kwargs.get('sample_count')
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("`sample_count` 必须是大于0的整数。")

    max_depth = kwargs.get('max_depth', 3)
    root_type = kwargs.get('root_type', 'dict') # 默认根类型为字典
    list_avg_len = kwargs.get('list_avg_len', 3)
    dict_avg_len = kwargs.get('dict_avg_len', 3)
    possible_leaf_types = kwargs.get('possible_leaf_types', ['int', 'float', 'str', 'bool'])

    if root_type not in ['list', 'dict']:
        raise ValueError("`root_type` 必须是 'list' 或 'dict'。")

    samples = []
    for _ in range(sample_count):
        if root_type == 'list':
            sample = [generate_random_value(random.choice(possible_leaf_types + ['list', 'dict']), 
                                            1, max_depth, list_avg_len, dict_avg_len) 
                      for _ in range(random.randint(1, list_avg_len * 2))]
        else: # root_type == 'dict'
            sample = {
                f"root_key_{i}": generate_random_value(random.choice(possible_leaf_types + ['list', 'dict']), 
                                                      1, max_depth, list_avg_len, dict_avg_len)
                for i in range(random.randint(1, dict_avg_len * 2))
            }
        samples.append(sample)
    return samples

if __name__ == "__main__":
    print("--- 示例 1: 生成 2 个最大深度为 2 的字典样本 ---")
    try:
        sample_data1 = generate_nested_sample_set(sample_count=2, max_depth=2, root_type='dict')
        for i, sample in enumerate(sample_data1):
            print(f"样本 {i+1}:\n{sample}\n")
    except ValueError as e:
        print(f"错误: {e}")

    print("\n--- 示例 2: 生成 3 个最大深度为 4 的列表样本，叶子类型只包含整数和浮点数 ---")
    try:
        sample_data2 = generate_nested_sample_set(sample_count=3, max_depth=4, root_type='list', 
                                                possible_leaf_types=['int', 'float'],
                                                list_avg_len=2, dict_avg_len=2)
        for i, sample in enumerate(sample_data2):
            print(f"样本 {i+1}:\n{sample}\n")
    except ValueError as e:
        print(f"错误: {e}")

    print("\n--- 示例 3: 生成 1 个只包含基本类型的字典样本 (max_depth=0 或 1) ---")
    try:
        sample_data3 = generate_nested_sample_set(sample_count=1, max_depth=1, root_type='dict',
                                                possible_leaf_types=['str', 'bool'])
        for i, sample in enumerate(sample_data3):
            print(f"样本 {i+1}:\n{sample}\n")
    except ValueError as e:
        print(f"错误: {e}")

    print("\n--- 示例 4: 错误示范 - 缺少 sample_count ---")
    try:
        generate_nested_sample_set(max_depth=2)
    except ValueError as e:
        print(f"预期错误: {e}")
