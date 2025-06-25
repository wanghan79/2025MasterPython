import random
import string


def random_sample(structure):
    if isinstance(structure, dict):
        return {k: random_sample(v) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [random_sample(structure[0]) for _ in range(structure[1])]
    elif structure == int:
        return random.randint(0, 100)
    elif structure == float:
        return random.uniform(0, 100)
    elif structure == str:
        length = random.randint(1, 10)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    elif structure == bool:
        return random.choice([True, False])
    else:
        return None

def generate_samples(sample_num=1, **kwargs):
    structure = kwargs.get('structure')
    if not structure:
        raise ValueError("Please provide a 'structure' keyword argument.")
    return [random_sample(structure) for item in range(sample_num)]

samples = generate_samples(
    sample_num=3,
    structure={
        'id': int,
        'name': str,
        'scores': [float, 5],
        'info': {
            'active': bool,
            'tags': [str, 3]
        }
    }
)
for item in samples:
    print(item)
