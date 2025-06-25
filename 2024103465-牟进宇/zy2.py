import random
import string

def generate_samples(**kwargs):
    # 解析参数
    structure = kwargs.get('structure')
    sample_count = kwargs.get('samples', 1)
    if not structure:
        raise ValueError("必须通过kwargs提供'structure'参数定义数据结构")

    # 递归生成单个样本的核心函数
    def _generate_structure(struct):
        type_map = {
            'int': _generate_int,
            'float': _generate_float,
            'str': _generate_str,
            'bool': _generate_bool,
            'list': _generate_list,
            'dict': _generate_dict,
            'tuple': _generate_tuple
        }
        data_type = struct.get('type')
        if data_type not in type_map:
            raise ValueError(f"不支持的类型: {data_type}（支持类型：{list(type_map.keys())}）")
        return type_map[data_type](struct)

    # 基础类型生成函数
    def _generate_int(struct):
        return random.randint(struct.get('min', 0), struct.get('max', 100))

    def _generate_float(struct):
        return random.uniform(struct.get('min', 0.0), struct.get('max', 1.0))

    def _generate_str(struct):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(struct.get('length', 5)))

    def _generate_bool(struct):
        return random.choice([True, False])

    # 容器类型生成函数（递归调用）
    def _generate_list(struct):
        if 'element' not in struct or 'length' not in struct:
            raise ValueError("list类型需要包含'length'（长度）和'element'（元素结构）字段")
        return [_generate_structure(struct['element']) for _ in range(struct['length'])]

    def _generate_dict(struct):
        if 'keys' not in struct:
            raise ValueError("dict类型需要包含'keys'（键值对结构）字段")
        return {key: _generate_structure(spec) for key, spec in struct['keys'].items()}

    def _generate_tuple(struct):
        if 'elements' not in struct:
            raise ValueError("tuple类型需要包含'elements'（元素结构列表）字段")
        return tuple(_generate_structure(elem_spec) for elem_spec in struct['elements'])

    # 生成指定数量的样本
    return [_generate_structure(structure) for _ in range(sample_count)]

# 示例用法
if __name__ == "__main__":
    # 定义嵌套结构（可根据需要修改）
    test_structure = {
        'type': 'list',
        'length': 3,
        'element': {
            'type': 'dict',
            'keys': {
                'id': {'type': 'int', 'min': 1, 'max': 1000},
                'name': {'type': 'str', 'length': 8},
                'scores': {
                    'type': 'tuple',
                    'elements': [
                        {'type': 'float', 'min': 0.0, 'max': 100.0},
                        {'type': 'float', 'min': 0.0, 'max': 100.0}
                    ]
                }
            }
        }
    }

    # 生成5个样本
    samples = generate_samples(structure=test_structure, samples=5)
    for idx, sample in enumerate(samples, 1):
        print(f"样本 {idx}: {sample}")
