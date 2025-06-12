import random
import string
from pprint import pprint

# --- 随机数据生成器 ---
def random_str(length=5):
    return ''.join(random.choices(string.ascii_letters, k=length))

def random_int():
    return random.randint(0, 100)

def random_float():
    return round(random.uniform(0, 100), 2)

def random_bool():
    return random.choice([True, False])

# --- 样本生成函数 ---
def generate_samples(**kwargs):
    structure = kwargs.get('structure')
    num = kwargs.get('num', 1)

    def generate_value(schema):
        if isinstance(schema, dict):
            return {key: generate_value(value) for key, value in schema.items()}
        elif isinstance(schema, (list, tuple, set)) and len(schema) == 1:
            container_type = type(schema)
            generated = [generate_value(schema[0]) for _ in range(random.randint(1, 3))]
            return container_type(generated)
        elif schema == int:
            return random_int()
        elif schema == float:
            return random_float()
        elif schema == str:
            return random_str()
        elif schema == bool:
            return random_bool()
        else:
            return None  # 不支持的类型

    return [generate_value(structure) for _ in range(num)]

# --- 示例调用 ---
if __name__ == "__main__":
    # 定义嵌套结构模板
    sample_structure = {
        "user": {
            "name": str,
            "age": int,
            "is_active": bool,
            "tags": [str],
        },
        "location": (float, float),
        "scores": [float]
    }

    # 生成 3 个样本
    samples = generate_samples(structure=sample_structure, num=3)

    # 打印结果
    pprint(samples)
