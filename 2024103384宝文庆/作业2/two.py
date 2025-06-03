import random

def generate_random_sample(**kwargs):
    def generate_sample(levels):
        if levels == 0:
            return random.randint(0, 100)
        else:
            return [generate_sample(levels - 1) for _ in range(random.randint(1, 5))]

    sample_structure = kwargs.get("structure", 2)
    sample_count = kwargs.get("count", 1)
    return [generate_sample(sample_structure) for _ in range(sample_count)]


sample = generate_random_sample(structure=3, count=5)
print(sample)
