import random
import string

def generate_random_value(data_type):
    
    if data_type == int:
        return random.randint(0, 100)
    elif data_type == float:
        return round(random.uniform(0, 100), 2)
    elif data_type == str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    elif data_type == bool:
        return random.choice([True, False])
    else:
        return None

def generate_sample(structure):
   
    if isinstance(structure, dict):
        return {k: generate_sample(v) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [generate_sample(v) for v in structure]
    elif isinstance(structure, tuple):
        return tuple(generate_sample(v) for v in structure)
    else:
        return generate_random_value(structure)

def generate_samples(**kwargs):
 
    structure = kwargs.get('structure')
    num = kwargs.get('num', 1)

    if structure is None:
        raise ValueError("缺少必要参数 'structure'")

    return [generate_sample(structure) for _ in range(num)]


if __name__ == "__main__":
    sample_structure = {
        'id': int,
        'name': str,
        'scores': [float, float, float],
        'active': bool,
        'profile': {
            'age': int,
            'gender': str,
            'tags': (str, str)
        }
    }

    samples = generate_samples(structure=sample_structure, num=5)
    for i, sample in enumerate(samples):
        print(f"样本 {i + 1}: {sample}")
