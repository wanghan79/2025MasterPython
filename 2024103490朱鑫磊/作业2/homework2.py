import random
import string

def generate_random_value(dtype):
    if dtype == int:
        return random.randint(0, 100)
    elif dtype == float:
        return round(random.uniform(0, 100), 2)
    elif dtype == str:
        return ''.join(random.choices(string.ascii_letters, k=5))
    elif dtype == bool:
        return random.choice([True, False])
    elif dtype == None:
        return None
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

def generate_sample(structure):
    if isinstance(structure, list):
        return [generate_sample(sub) for sub in structure]
    elif isinstance(structure, tuple):
        return tuple(generate_sample(sub) for sub in structure)
    elif isinstance(structure, set):
        return set(generate_sample(sub) for sub in structure)
    elif isinstance(structure, dict):
        return {k: generate_sample(v) for k, v in structure.items()}
    elif isinstance(structure, type):
        return generate_random_value(structure)
    else:
        raise TypeError(f"Invalid structure type: {type(structure)}")

def generate_samples(**kwargs):
    structure = kwargs.get("structure")
    num_samples = kwargs.get("num_samples", 1)
    if structure is None:
        raise ValueError("You must provide a 'structure' keyword argument.")
    return [generate_sample(structure) for _ in range(num_samples)]

if __name__ == "__main__":
    structure_template = {
        "id": int,
        "info": {
            "name": str,
            "scores": [float, float, float],
            "tags": {"tag1": str, "tag2": str}
        },
        "flags": (bool, bool),
    }

    samples = generate_samples(structure=structure_template, num_samples=5)
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}:\n{sample}\n")
