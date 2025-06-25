import random
import string
from datetime import datetime, timedelta
from typing import Any


def random_int():
    return random.randint(0, 1000)


def random_float():
    return round(random.uniform(0, 1000), 2)


def random_str(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def random_bool():
    return random.choice([True, False])


def random_date(start_year=2000, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return (start + timedelta(days=random.randint(0, (end - start).days))).strftime("%Y-%m-%d")


# 样本构造器
def generate_sample(template: Any) -> Any:
    """
    递归生成与 template 匹配的随机数据
    """
    if isinstance(template, list):
        return [generate_sample(template[0]) for _ in range(random.randint(1, 3))]
    elif isinstance(template, tuple):
        return tuple(generate_sample(item) for item in template)
    elif isinstance(template, dict):
        return {key: generate_sample(val) for key, val in template.items()}
    elif template == "int":
        return random_int()
    elif template == "float":
        return random_float()
    elif template == "str":
        return random_str()
    elif template == "bool":
        return random_bool()
    elif template == "date":
        return random_date()
    else:
        return None


def DataSampler(num: int = 5, **kwargs) -> list:
    """
    DataSampler(样本数量, data=结构描述模板)

    示例：
    >>> DataSampler(3, data={"id": "int", "name": "str", "scores": ["float"], "birthday": "date"})
    """
    if "data" not in kwargs:
        raise ValueError("必须通过参数 data=... 提供数据结构模板")

    structure_template = kwargs["data"]
    return [generate_sample(structure_template) for _ in range(num)]


# 示例用法
if __name__ == "__main__":
    template = {
        "id": "int",
        "name": "str",
        "scores": ["float"],
        "info": {
            "gender": "str",
            "vip": "bool",
            "birth": "date"
        },
        "tags": ("str", "str")
    }

    samples = DataSampler(3, data=template)
    for idx, sample in enumerate(samples):
        print(f"样本 {idx + 1}:")
        print(sample)
        print()
