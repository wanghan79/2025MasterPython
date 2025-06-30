import random
import string
from datetime import datetime, timedelta


def DataSampler(n, **kwargs):
    def generate_value(spec):
        if isinstance(spec, type):
            if spec == int: return random.randint(1, 10000)
            if spec == float: return round(random.uniform(1.0, 1000.0), 2)
            if spec == str: return ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            if spec == bool: return random.choice([True, False])
            if spec == datetime:
                start = datetime(2000, 1, 1)
                end = datetime.now()
                return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))
            raise ValueError(f"Unsupported type: {spec}")
        if isinstance(spec, list): return [generate_value(spec[0]) for _ in range(spec[1])]
        if isinstance(spec, tuple): return tuple(generate_value(item) for item in spec)
        if isinstance(spec, dict): return {k: generate_value(v) for k, v in spec.items()}
        raise ValueError(f"Invalid specification: {spec}")

    samples = []
    for _ in range(n):
        sample = {}
        for key, spec in kwargs.items():
            sample[key] = generate_value(spec)
        samples.append(sample)
    return samples


if __name__ == "__main__":
    data_spec = {
        "id": int,
        "name": str,
        "active": bool,
        "scores": [float, 5],
        "details": {
            "created": datetime,
            "tags": (str, str, str),
            "metadata": [int, 3]
        }
    }

    samples = DataSampler(3, **data_spec)
    for sample in samples:
        print(sample)
