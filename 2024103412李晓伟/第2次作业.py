import random
import string
import copy
from datetime import datetime, date

def generate_random_sample(**kwargs):
    def generate_random_string(length=5):
        return ''.join(random.choice(string.ascii_letters) for _ in range(length))

    def generate_random_int():
        return random.randint(1, 100)

    def generate_random_float():
        return round(random.uniform(1.0, 100.0), 2)

    def generate_random_bool():
        return random.choice([True, False])

    def generate_random_date():
        return date.today()

    def generate_random_list(length=3, value_type='int'):
        return [generate_random_value(value_type) for _ in range(length)]

    def generate_random_dict(keys, values):
        return {key: generate_random_value(value) for key, value in zip(keys, values)}

    def generate_random_value(value_type):
        if value_type == 'str':
            return generate_random_string()
        elif value_type == 'int':
            return generate_random_int()
        elif value_type == 'float':
            return generate_random_float()
        elif value_type == 'bool':
            return generate_random_bool()
        elif value_type == 'date':
            return generate_random_date()
        elif value_type == 'list':
            return generate_random_list()
        elif isinstance(value_type, dict):
            return generate_random_dict(value_type.keys(), value_type.values())
        elif isinstance(value_type, list):
            return [generate_random_value(item) for item in value_type]
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

    data_structure = kwargs.get('data_structure')
    num_samples = kwargs.get('num_samples', 1)

    samples = []
    for i in range(num_samples):
        sample = generate_random_value(copy.deepcopy(data_structure))
        samples.append(sample)

    return samples

# 调用
data_structure = {
    'name': 'str',
    'age': 'int',
    'scores': ['int', 'int', 'int'],
    'is_male': 'bool',
    'created_time': 'date',
    'address': {
        'street': 'str',
        'number': 'int'
    }
}
num_samples = 6

samples = generate_random_sample(data_structure=data_structure, num_samples=num_samples)
for i, sample in enumerate(samples, 1):
    print(f"Sample {i}: {sample}")

# 输出：
# Sample 1: {'name': 'aVlAS', 'age': 95, 'scores': [45, 77, 32], 'is_male': False, 'created_time': datetime.date(2025, 6, 10), 'address': {'street': 'lOURS', 'number': 97}}
# Sample 2: {'name': 'mREne', 'age': 17, 'scores': [100, 8, 87], 'is_male': True, 'created_time': datetime.date(2025, 6, 10), 'address': {'street': 'lCFbn', 'number': 73}}
# Sample 3: {'name': 'HPgDC', 'age': 6, 'scores': [85, 28, 59], 'is_male': False, 'created_time': datetime.date(2025, 6, 10), 'address': {'street': 'wetEA', 'number': 76}}
# Sample 4: {'name': 'TfvHe', 'age': 37, 'scores': [59, 22, 55], 'is_male': False, 'created_time': datetime.date(2025, 6, 10), 'address': {'street': 'PCyrh', 'number': 91}}
# Sample 5: {'name': 'ORRUa', 'age': 55, 'scores': [65, 12, 29], 'is_male': True, 'created_time': datetime.date(2025, 6, 10), 'address': {'street': 'yDcSe', 'number': 1}}
# Sample 6: {'name': 'giQUH', 'age': 51, 'scores': [1, 66, 79], 'is_male': True, 'created_time': datetime.date(2025, 6, 10), 'address': {'street': 'Nsnen', 'number': 6}}

