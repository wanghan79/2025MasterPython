import random
import string
from typing import Any


def random_modify_data_structure(data: Any, depth: int = 4) -> Any:
    if depth == 0:
        return random.choice([
            random.randint(-100, 100),
            random.uniform(-100, 100),
            ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 15)))
        ])

    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if random.choice([True, False]):
                new_key = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 10)))
            else:
                new_key = key
            new_dict[new_key] = random_modify_data_structure(value, depth - 1)
        return new_dict

    elif isinstance(data, list):
        new_list = []
        for item in data:
            if random.choice([True, False]):
                new_list.append(random_modify_data_structure(item, depth - 1))
            else:
                new_list.append(item)
        return new_list

    elif isinstance(data, (str, int, float)):
        if random.choice([True, False]):
            return random.choice([
                random.randint(-100, 100),
                random.uniform(-100, 100),
                ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 15)))
            ])
        else:
            return data

    elif hasattr(data, '__dict__'):
        # 关键修改：绕过 __init__，直接通过 __dict__ 复制属性
        new_obj = data.__class__.__new__(data.__class__)
        new_obj.__dict__ = {
            key: random_modify_data_structure(value, depth - 1)
            for key, value in data.__dict__.items()
        }
        return new_obj

    else:
        return data


# 示例自定义类
class Person:
    def __init__(self, name: str, age: int, address: dict):
        self.name = name
        self.age = age
        self.address = address

    def __repr__(self):
        return f"Person(name={self.name}, age={self.age}, address={self.address})"


# 示例输入
#input_data = {
#    "person": Person(name="Bob", age=25, address={"city": "Dreamland", "zipcode": [54321, 98765]})
#}

class Address:
    def __init__(self, city: str, details: dict):
        self.city = city
        self.details = details  # 第三层字典，包含第四层数据

    def __repr__(self):
        return f"Address(city={self.city}, details={self.details})"

class Person:
    def __init__(self, name: str, age: int, address: Address):
        self.name = name
        self.age = age
        self.address = address  # 第二层自定义对象

    def __repr__(self):
        return f"Person(name={self.name}, age={self.age}, address={self.address})"

# 深度为4的输入数据
input_data = {
    "metadata": {  # 第一层字典
        "id": 1001,
        "tags": ["user", "premium"],  # 第二层列表
        "history": [  # 第二层列表（包含第三层字典）
            {
                "date": "2023-01-01",
                "action": "login",
                "details": {  # 第三层字典
                    "ip": "192.168.1.1",  # 第四层字符串
                    "location": Address(  # 第四层自定义对象
                        city="Paris",
                        details={
                            "lat": 48.8566,  # 第四层浮点数
                            "lon": 2.3522    # 第四层浮点数
                        }
                    )
                }
            }
        ]
    },
    "primary_user": Person(  # 第二层自定义对象
        name="Alice",
        age=30,
        address=Address(  # 第三层自定义对象
            city="London",
            details={
                "postcodes": ["SW1A", "NW1"],  # 第四层列表
                "population": 8900000          # 第四层整数
            }
        )
    )
}

# 打印数据结构
print("原始输入数据:")
print(input_data)

# 运行测试
modified_data = random_modify_data_structure(input_data)
print(modified_data)
