import random
import string

def random_students(structure):
    if isinstance(structure, dict):
        return {k: random_students(v) for k, v in structure.items()}
    elif isinstance(structure, list):
        if len(structure) == 0:
            return []
        elem_struct = structure[0]
        list_len = 3
        return [random_students(elem_struct) for _ in range(list_len)]
    elif structure == int:
        return random.randint(0, 100)
    elif structure == float:
        return random.uniform(0, 100)
    elif structure == str:
        length = random.randint(3, 10)
        return ''.join(random.choices(string.ascii_letters, k=length))
    elif structure == bool:
        return random.choice([True, False])
    else:
        return None

def generate_samples(sample_num=1, **kwargs):
    structure = kwargs.get('structure')
    if structure is None:
        raise ValueError("结构err")
    return [random_students(structure) for _ in range(sample_num)]

# 示例用法
if __name__ == "__main__":
    student_struct = {
        'name': str,
        'scores': [float],
        'profile': {
            'active': bool,
            'roommate': [str],
            'teachers': str,
        }
    }
    samples = generate_samples(sample_num=1, structure=student_struct)
    for s in samples:
        print(s)
# {
#       'name': 'ZvY', 
#       'scores': [94.65900932204278, 65.5074108532508, 93.4658604002919], 
#       'profile': {
#         'active': False, 
#         'roommate': ['hUJghkf', 'fPbwW', 'dojrZfdoEm'], 
#         'teachers': 'MlWCINyc'}
# }
