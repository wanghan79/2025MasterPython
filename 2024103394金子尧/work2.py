
import random
import string
import datetime
from typing import Any, Union

def random_str(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_date(start_year=2000, end_year=2023):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = (end - start).days
    return (start + datetime.timedelta(days=random.randint(0, delta))).isoformat()

def generate_field(data_type: Any) -> Any:
    """根据字段定义生成一个随机值"""
    if isinstance(data_type, dict):
        return {k: generate_field(v) for k, v in data_type.items()}
    elif isinstance(data_type, list):
        return [generate_field(data_type[0]) for _ in range(random.randint(1, 3))]
    elif isinstance(data_type, tuple):
        return tuple(generate_field(t) for t in data_type)
    elif data_type == int:
        return random.randint(0, 100)
    elif data_type == float:
        return round(random.uniform(0, 100), 2)
    elif data_type == str:
        return random_str()
    elif data_type == bool:
        return random.choice([True, False])
    elif data_type == 'date':
        return random_date()
    else:
        return None  # 不支持的类型

def data_sampler(schema: dict, **kwargs):
    """生成结构化随机数据样本"""
    num = kwargs.get("num", 1)
    return [generate_field(schema) for _ in range(num)]

# 示例用法
if __name__ == "__main__":
    # 定义嵌套数据结构
    sample_schema = {
        "id": int,
        "name": str,
        "active": bool,
        "signup_date": "date",
        "profile": {
            "age": int,
            "height": float,
            "tags": [str],
            "location": (float, float)
        },
        "login_history": [
            {
                "login_time": "date",
                "ip": str,
                "success": bool
            }
        ]
    }

    # 生成数据样本
    samples = data_sampler(sample_schema, num=5)

    # 输出查看
    from pprint import pprint
    pprint(samples)
