import random
import string


def generate_random_sample(nest_level=1, sample_size=1, data_types=None, **kwargs):
    """
    生成嵌套数据结构的随机样本。

    :param nest_level: 嵌套层级，控制数据的嵌套深度
    :param sample_size: 每个嵌套层级的样本数量
    :param data_types: 数据类型字典，指定每个层级的数据类型，如 {'int': (0, 100), 'str': 5}
    :param kwargs: 额外的参数，支持定制样本结构和样本个数
    :return: 生成的嵌套样本集
    """

    # 如果没有指定数据类型，则默认为整数范围
    if data_types is None:
        data_types = {'int': (0, 100)}  # 默认生成随机整数

    def generate_random_data(data_type):
        """根据给定的数据类型生成随机数据"""
        if data_type == 'int':
            return random.randint(*kwargs.get('int_range', (0, 100)))
        elif data_type == 'float':
            return random.uniform(*kwargs.get('float_range', (0, 100.0)))
        elif data_type == 'str':
            length = kwargs.get('str_length', 5)
            return ''.join(random.choices(string.ascii_letters, k=length))
        elif data_type == 'bool':
            return random.choice([True, False])
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def generate_nested_sample(level, data_types, sample_size):
        """递归生成嵌套数据结构"""
        if level == 0:  # 达到最底层，不再嵌套
            # 随机选择数据类型
            data_type = random.choice(list(data_types.keys()))
            # 直接将数据类型的随机值放入列表中
            return [generate_random_data(data_type) for _ in range(sample_size)]
        else:
            # 递归生成下层嵌套
            sample = []
            for _ in range(sample_size):
                nested_sample = {}
                for data_type in data_types:
                    # 将每个数据类型直接生成随机值而不是再嵌套字典
                    nested_sample[data_type] = generate_nested_sample(level - 1, {data_type: data_types[data_type]}, 1)
                sample.append(nested_sample)
            return sample

    # 主函数开始生成数据
    return generate_nested_sample(nest_level, data_types, sample_size)


# 测试函数
if __name__ == "__main__":
    # 生成一个有2层嵌套结构的样本
    result = generate_random_sample(
        nest_level=2,
        sample_size=3,
        data_types={'int': (0, 100), 'str': 5, 'float': (0, 100.0)},
        int_range=(0, 50),
        str_length=8,
        float_range=(10.0, 50.0)
    )

    for item in result:
        print(item)
