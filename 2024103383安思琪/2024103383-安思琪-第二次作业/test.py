import random
import string

def create_random_object(**kwargs):

    random_object = {}

    for key, value in kwargs.items():
        if value == int:
            random_object[key] = random.randint(0, 100)
        elif value == float:
            random_object[key] = random.uniform(0, 100)
        elif value == str:
            random_object[key] = ''.join(random.choices(string.ascii_letters, k=10))
        elif value == bool:
            random_object[key] = random.choice([True, False])
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            if all(isinstance(x, int) for x in value):
                random_object[key] = random.randint(value[0], value[1])
            elif all(isinstance(x, float) for x in value):
                random_object[key] = random.uniform(value[0], value[1])
            else:
                random_object[key] = random.choice(value)
        else:
            random_object[key] = value

    return random_object


random_obj = create_random_object(
    age=int,
    height=float,
    name=str,
    is_student=bool,
    score=(0, 100),
    favorite_colors=("red", "blue", "green")
)

print(random_obj)