import random
import string

def generate_nested_samples(**kwargs):
    count = kwargs.get('count', 1)
    structure = {k: v for k, v in kwargs.items() if k != 'count'}
    return [_generate_sample(structure) for _ in range(count)]

def _generate_sample(structure):
    result = {}
    
    for field_name, field_type in structure.items():
        if field_type == 'int':
            result[field_name] = random.randint(1, 100)
        elif field_type == 'float':
            result[field_name] = round(random.uniform(1.0, 100.0), 2)
        elif field_type == 'str':
            result[field_name] = ''.join(random.choices(string.ascii_letters, k=5))
        elif field_type == 'bool':
            result[field_name] = random.choice([True, False])
        elif isinstance(field_type, dict):
            result[field_name] = _generate_sample(field_type)
        elif isinstance(field_type, list) and field_type:
            item_type = field_type[0]
            length = random.randint(2, 4)
            if isinstance(item_type, dict):
                result[field_name] = [_generate_sample(item_type) for _ in range(length)]
            elif item_type == 'int':
                result[field_name] = [random.randint(1, 100) for _ in range(length)]
            elif item_type == 'float':
                result[field_name] = [round(random.uniform(1.0, 100.0), 2) for _ in range(length)]
            elif item_type == 'str':
                result[field_name] = [''.join(random.choices(string.ascii_letters, k=4)) for _ in range(length)]
    
    return result

def test():
    print("嵌套数据类型样本生成测试")
    
    samples = generate_nested_samples(
        count=2,
        id='int',
        name='str',
        user_info={'age': 'int', 'email': 'str'},
        scores=['float']
    )
    
    for i, sample in enumerate(samples, 1):
        print(f"样本{i}: {sample}")

if __name__ == "__main__":
    test()