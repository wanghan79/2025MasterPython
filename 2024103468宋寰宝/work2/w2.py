import random

import string


def generate_random_string(length):
    all_characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choice(all_characters) for _ in range(
        length))
    return random_string


def generate_data(**kwargs):
    ds = kwargs['collection']['constructor']()
    data_type = kwargs['type']
    add = kwargs['collection']['add']
    if data_type == 'sub_items':
        for si in kwargs['sub_items']:
            add(ds, generate_data(**si))
    else:
        for _ in range(kwargs['collection']['len']):
            add(ds, kwargs['generator']())
    return ds


if __name__ == '__main__':
    kw = {
        "type": "sub_items",
        "collection": {
            "constructor": list,
            "add": lambda ds, item: ds.append(item)
        },
        "sub_items": [
            {
                "type": "sub_items",
                "sub_items": [
                    {
                        "type": "generator",
                        "collection": {
                            "constructor": list,
                            "add": lambda ds, item: ds.append(item),
                            "len": 5
                        },
                        "generator": lambda: ((random.randint(0, 255),
                                              random.randint(0, 255),
                                              random.randint(0, 255))),
                    }
                ],
                "collection": {
                    "constructor": list,
                    "add": lambda ds, item: ds.append(item),
                },
            },
            {
                "type": "generator",
                "generator": lambda: random.randint(0, 10),
                "collection": {
                    "constructor": list,
                    "add": lambda ds, item: ds.append(item),
                    "len": 10,
                },
            },
            {
                "type": "generator",
                "generator": lambda: random.uniform(0., 10.),
                "collection": {
                    "constructor": list,
                    "add": lambda ds, item: ds.append(item),
                    "len": 10,
                },
            },
            {
                "type": "generator",
                "generator": lambda: generate_random_string(10),
                "collection": {
                    "constructor": list,
                    "add": lambda ds, item: ds.append(item),
                    "len": 10,
                },
            }
        ],

    }
    print(generate_data(**kw))
