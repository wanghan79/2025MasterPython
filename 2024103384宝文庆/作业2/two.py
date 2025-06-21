import random
import string
import datetime

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_date(start_year=2000, end_year=2025):
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    delta = end_date - start_date
    rand_days = random.randint(0, delta.days)
    return start_date + datetime.timedelta(days=rand_days)

def generate_sample(structure):
    if isinstance(structure, dict):
        return {k: generate_sample(v) for k, v in structure.items()}
    elif isinstance(structure, list):
        return [generate_sample(structure[0]) for _ in range(random.randint(1, 3))]
    elif isinstance(structure, tuple):
        return tuple(generate_sample(v) for v in structure)
    elif structure == int:
        return random.randint(0, 100)
    elif structure == float:
        return round(random.uniform(0, 100), 2)
    elif structure == str:
        return random_string()
    elif structure == bool:
        return random.choice([True, False])
    elif structure == datetime.date:
        return random_date()
    else:
        return None

def DataSampler(**kwargs):
    count = kwargs.pop('count', 5)
    structure = kwargs.get('structure')
    return [generate_sample(structure) for _ in range(count)]

structure = {
    "id": int,
    "name": str,
    "price": float,
    "scores": [float],
    "num": [int],
    "is_active": bool,
    "birth": datetime.date,
    "map": (float,float),
    "profile": {
        "height": float,
        "weight": float,
        "tags": (str, str)
    }
}
samples = DataSampler(structure=structure, count=3)
from pprint import pprint
pprint(samples)
