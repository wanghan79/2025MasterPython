import random
import string
from typing import Any, Union

class SampleGenerator:
    def __init__(self, structure: Any):
        """
        初始化样本生成器
        :param structure: 嵌套结构模板，如 {'name': str, 'age': int, 'tags': [str]}
        """
        self.structure = structure

    def _generate_random_str(self, length=6) -> str:
        return ''.join(random.choices(string.ascii_letters, k=length))

    def _generate_value(self, spec: Any) -> Any:
        if isinstance(spec, type):
            if spec == int:
                return random.randint(0, 100)
            elif spec == float:
                return round(random.uniform(0, 100), 2)
            elif spec == str:
                return self._generate_random_str()
            elif spec == bool:
                return random.choice([True, False])
            else:
                return None
        elif isinstance(spec, list) and spec:
            # 默认生成固定长度（3）列表，内部类型为 spec[0]
            return [self._generate_value(spec[0]) for _ in range(3)]
        elif isinstance(spec, tuple) and spec:
            return tuple(self._generate_value(item) for item in spec)
        elif isinstance(spec, dict):
            return {k: self._generate_value(v) for k, v in spec.items()}
        else:
            return None

    def generate_one(self) -> Any:
        """生成一个样本"""
        return self._generate_value(self.structure)

    def generate(self, count: int) -> list:
        """生成多个样本"""
        return [self.generate_one() for _ in range(count)]


# 示例测试
if __name__ == "__main__":
    sample_structure = {
        "name": str,
        "age": int,
        "scores": [float],
        "profile": {
            "height": float,
            "active": bool
        },
        "tags": (str, str)
    }

    gen = SampleGenerator(sample_structure)
    samples = gen.generate(5)
    for i, s in enumerate(samples, 1):
        print(f"样本 {i}：{s}")
