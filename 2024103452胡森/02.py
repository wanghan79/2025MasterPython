from datetime import date
import random
import string


class DataSampler:
    def __init__(self, **kwargs):
        self.count = kwargs['count']
        self.structure = {k: v for k, v in kwargs.items() if k != 'count'}

    def __call__(self):
        def sample(item):
            if isinstance(item, dict):
                return {k: sample(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [sample(i) for i in item]
            elif item == int:
                return random.randint(0, 65535)
            elif item == float:
                return random.uniform(0, 7355608)
            elif item == date:
                return date.today().replace(day=random.randint(1, 28), month=random.randint(1, 12), year=random.randint(2000, 2077))
            elif item == str:
                length = random.randint(1, 10)
                return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            elif item == bool:
                return random.choice([True, False])
            else:
                return None
        return [sample(self.structure) for _ in range(self.count)]


if __name__ == "__main__":
    datasampler = DataSampler(
        count=2,
        id=int,
        name=str,
        user_info={'age': int, 'email': str},
        date=date,
        scores=[float]
    )
    res = datasampler()
    print(res)

# result:
# [{'id': 38141, 'name': 'RamSnjaDep', 'user_info': {'age': 1831, 'email': '8QNl'}, 'date': datetime.date(2000, 7, 22), 'scores': [4994868.766768865]}, 
# {'id': 37503, 'name': 'PKea0s8eht', 'user_info': {'age': 62368, 'email': 'G6k'}, 'date': datetime.date(2029, 8, 11), 'scores': [3484907.6577074844]}]
