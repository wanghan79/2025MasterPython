import random
import string
from typing import Any
from copy import deepcopy


def random_data_generator(data: Any,
                          mutation_rate: float = 0.3,
                          max_depth: int = 3) -> Any:
    """
    递归随机化处理嵌套数据结构

    :param data: 输入数据（支持任意嵌套结构）
    :param mutation_rate: 每个节点的变异概率 (0~1)
    :param max_depth: 最大处理嵌套深度
    :return: 随机化处理后的新数据结构
    """

    def process_primitive(value: Any) -> Any:
        """处理基础数据类型"""
        if random.random() > mutation_rate:
            return value

        if isinstance(value, str):
            if len(value) > 0:
                ops = random.choice(['delete', 'replace', 'insert'])
                pos = random.randint(0, len(value) - 1)
                if ops == 'delete':
                    return value[:pos] + value[pos + 1:]
                elif ops == 'replace':
                    new_char = random.choice(string.ascii_letters + string.digits)
                    return value[:pos] + new_char + value[pos + 1:]
                else:
                    new_char = random.choice(string.printable)
                    return value[:pos] + new_char + value[pos:]
            return value

        elif isinstance(value, (int, float)):
            return value * random.uniform(0.9, 1.1)

        elif isinstance(value, bool):
            return not value

        return value

    def process_container(container: Any, depth: int) -> Any:
        """递归处理容器类型"""
        if depth > max_depth:
            return container

        if isinstance(container, dict):
            new_dict = {}
            for k, v in container.items():
                if random.random() < mutation_rate / 2:
                    continue
                new_key = process_primitive(k) if random.random() < mutation_rate else k
                new_val = random_data_generator(v, mutation_rate, depth + 1)
                new_dict[new_key] = new_val

            if random.random() < mutation_rate:
                new_key = ''.join(random.choices(string.ascii_letters, k=3))
                new_dict[new_key] = random_data_generator(None, mutation_rate, depth + 1)
            return new_dict

        elif isinstance(container, (list, tuple, set)):
            temp_list = list(deepcopy(container))

            if len(temp_list) > 0 and random.random() < mutation_rate:
                del temp_list[random.randint(0, len(temp_list) - 1)]

            if random.random() < mutation_rate:
                temp_list.insert(random.randint(0, len(temp_list)),
                                 random_data_generator(None, mutation_rate, depth + 1))

            processed = [random_data_generator(item, mutation_rate, depth + 1)
                         for item in temp_list]

            if isinstance(container, (list, set)) and random.random() < mutation_rate:
                random.shuffle(processed)

            if isinstance(container, tuple):
                return tuple(processed)
            elif isinstance(container, set):
                return set(processed)
            return processed

        return container

    if data is None:
        return random.choice([
            random.randint(-100, 100),
            ''.join(random.choices(string.ascii_letters, k=5)),
            random.random()
        ])

    if not isinstance(data, (dict, list, tuple, set)):
        return process_primitive(data)

    return process_container(data, depth=1)


# 测试用例
if __name__ == "__main__":
    complex_data = {
        "text": ["Hello", {"nested": ("world", [42, True])}],
        "matrix": [[[1, 2], (3, 4)], {"a": 5.5, "b": None}],
        "metadata": {"author": "GPT-3", "score": 95}
    }

    randomized = random_data_generator(complex_data)
    print("Original:", complex_data)
    print("Randomized:", randomized)
