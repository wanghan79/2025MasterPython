import random
import string

class DataSampler:
    def __init__(self, structure: dict):
        self.structure = structure

    def sample(self, count=10):
        return [self._generate_one() for _ in range(count)]

    def _generate_one(self):
        result = {}
        for key, dtype in self.structure.items():
            result[key] = self._generate_value(dtype)
        return result

    def _generate_value(self, dtype):
        if dtype == int:
            return random.randint(0, 100)
        elif dtype == float:
            return round(random.uniform(0, 100), 2)
        elif dtype == str:
            return ''.join(random.choices(string.ascii_letters, k=5))
        elif dtype == bool:
            return random.choice([True, False])
        elif dtype == list:
            return [random.randint(0, 10) for _ in range(3)]
        elif dtype == tuple:
            return tuple(random.randint(0, 10) for _ in range(3))
        elif dtype == dict:
            return {f"k{i}": random.randint(0, 10) for i in range(2)}
        else:
            return None

if __name__ == "__main__":
    structure = {
        "id": int,
        "score": float,
        "name": str,
        "active": bool
    }

    sampler = DataSampler(structure)
    data = sampler.sample(20)
    print("生成的数据：", data)

