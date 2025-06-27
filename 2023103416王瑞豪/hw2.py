import random
import string
from typing import Any, Union

# 辅助函数：生成随机字符串
def random_str(length=6):
    return ''.join(random.choices(string.ascii_letters, k=length))

# 样本生成函数：支持任意嵌套结构
def generate_sample(structure: Any) -> Any:
    if isinstance(structure, type):
        if structure == int:
            return random.randint(0, 100)
        elif structure == float:
            return round(random.uniform(0, 100), 2)
        elif structure == str:
            return random_str()
        elif structure == bool:
            return random.choice([True, False])
        else:
            return None  # 不支持的基础类型
    elif isinstance(structure, list) and structure:
        return [generate_sample(structure[0]) for _ in range(random.randint(1, 3))]
    elif isinstance(structure, tuple) and structure:
        return tuple(generate_sample(item) for item in structure)
    elif isinstance(structure, dict):
        return {key: generate_sample(val) for key, val in structure.items()}
    else:
        return None

# 主函数：生成多个样本
def generate_samples(**kwargs):
    count = kwargs.get('count', 1)
    structure = kwargs.get('structure', {})
    return [generate_sample(structure) for _ in range(count)]


structure_template = {
        "id": int,
        "info": {
            "name": str,
            "scores": [float, float, float],
            "tags": {"tag1": str, "tag2": str}
        },
        "flags": (bool, bool),
    }

samples = generate_samples(structure=structure_template, num_samples=5)
for i, sample in enumerate(samples, 1):
    print(f"Sample {i}:\n{sample}\n")