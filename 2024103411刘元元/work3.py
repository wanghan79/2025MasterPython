import random
import string
from datetime import datetime, timedelta
import numpy as np
from functools import wraps


# 定义统计修饰器
def stats_decorator(*stats):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 调用原始方法
            result = func(self, *args, **kwargs)

            # 获取所有叶节点并筛选出 int 或 float 类型的数据
            leaf_nodes = self.get_leaf_nodes(result)
            numeric_leaf_nodes = [leaf for leaf in leaf_nodes if isinstance(leaf['value'], (int, float))]

            # 初始化统计结果字典
            stats_result = {}

            # 按指定的统计量进行计算
            numeric_values = [leaf['value'] for leaf in numeric_leaf_nodes]
            if numeric_values:  # 只有存在数值型数据时才计算统计量
                for stat in stats:
                    if stat == 'mean':
                        stats_result[stat] = np.mean(numeric_values)
                    elif stat == 'variance':
                        stats_result[stat] = np.var(numeric_values, ddof=0)  # 总体方差
                    elif stat == 'rmse':
                        stats_result[stat] = np.sqrt(np.var(numeric_values, ddof=0))
                    elif stat == 'sum':
                        stats_result[stat] = np.sum(numeric_values)

            # 返回原始结果和统计结果
            return result, stats_result

        return wrapper
    return decorator


class DataSampler:
    def __init__(self, random_generator=None):
        """
        初始化数据采样器。

        Args:
            random_generator: 可选的随机数生成器，默认为random.SystemRandom
        """
        self.random_generator = random_generator or random.SystemRandom()

    def generate_value(self, value_def):
        """
        根据值定义生成相应的随机值。

        Args:
            value_def (dict): 包含数据类型和参数的字典。

        Returns:
            生成的随机值。
        """
        if "type" not in value_def:
            return value_def  # 如果没有指定类型，直接返回值

        data_type = value_def["type"]
        params = {k: v for k, v in value_def.items() if k != "type"}

        if data_type == "int":
            return self.random_generator.randint(params["datarange"][0], params["datarange"][1])
        elif data_type == "float":
            return self.random_generator.uniform(params["datarange"][0], params["datarange"][1])
        elif data_type == "str":
            length = params.get("len", 10)
            return ''.join(self.random_generator.choice(params["datarange"]) for _ in range(length))
        elif data_type == "bool":
            return self.random_generator.choice([True, False])
        elif data_type == "date":
            start = datetime.strptime(params["datarange"][0], "%Y-%m-%d")
            end = datetime.strptime(params["datarange"][1], "%Y-%m-%d")
            delta = end - start
            random_days = self.random_generator.randint(0, delta.days)
            return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")
        elif data_type == "list":
            list_length = params.get("len", 3)
            element_def = params["elements"]
            return [self.generate_value(element_def) for _ in range(list_length)]
        elif data_type == "tuple":
            tuple_length = params.get("len", 3)
            element_def = params["elements"]
            return tuple(self.generate_value(element_def) for _ in range(tuple_length))
        elif data_type == "dict":
            return {k: self.generate_value(v) for k, v in params["subs"].items()}
        else:
            return None

    def generate(self, **kwargs):
        """
        根据给定的参数生成相应的数据。

        Args:
            **kwargs: 包含数据类型、范围和结构的键值对。

        Returns:
            生成样本数据字典。
        """
        sample_dict = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                sample_dict[key] = self.generate_value(value)
            else:
                sample_dict[key] = value
        return sample_dict

    @stats_decorator('mean', 'variance', 'rmse', 'sum')
    def analyze(self, data):
        # 这里可以添加对数据的分析逻辑，目前先直接返回数据
        return data

    def get_leaf_nodes(self, data):
        """
        递归获取数据结构中的所有叶节点，并返回包含叶节点信息的列表。

        Args:
            data: 数据结构（可以是 dict、list 或基本类型）。

        Returns:
            list: 包含叶节点信息的列表，每个叶节点信息是一个字典，包含 'path' 和 'value'。
        """
        leaf_nodes = []

        def traverse(value, path=""):
            if isinstance(value, dict):
                for k, v in value.items():
                    traverse(v, f"{path}.{k}" if path else k)
            elif isinstance(value, list) or isinstance(value, tuple):
                for idx, item in enumerate(value):
                    traverse(item, f"{path}[{idx}]")
            else:
                leaf_nodes.append({"path": path, "value": value})

        traverse(data)
        return leaf_nodes


if __name__ == "__main__":
    # 定义数据结构和类型，使用自定义键名
    data_structure = {
        "user_id": {
            "type": "int",
            "datarange": (100000, 999999)
        },
        "is_active": {
            "type": "bool"
        },
        "registration_date": {
            "type": "date",
            "datarange": ("2020-01-01", "2023-12-31")
        },
        "profile": {
            "type": "dict",
            "subs": {
                "age": {
                    "type": "int",
                    "datarange": (18, 99)
                },
                "gender": {
                    "type": "str",
                    "datarange": ["Male", "Female"],
                    "len": 1
                }
            }
        },
        "accounts": {
            "type": "list",
            "len": 3,
            "elements": {
                "type": "dict",
                "subs": {
                    "account_id": {
                        "type": "int",
                        "datarange": (1000000, 9999999)
                    },
                    "balance": {
                        "type": "float",
                        "datarange": (0.0, 1000000.0)
                    },
                    "transactions": {
                        "type": "tuple",
                        "len": 5,
                        "elements": {
                            "type": "float",
                            "datarange": (-10000.0, 10000.0)
                        }
                    }
                }
            }
        },
        "activity_log": {
            "type": "dict",
            "subs": {
                "login_attempts": {
                    "type": "int",
                    "datarange": (0, 20)
                }
            }
        },
        "device_info": {
            "type": "dict",
            "subs": {
                "device_id": {
                    "type": "str",
                    "datarange": string.ascii_uppercase + string.digits,
                    "len": 12
                },
                "browser": {
                    "type": "str",
                    "datarange": ["Chrome", "Firefox", "Safari", "Edge", "Opera"],
                    "len": 1
                },
                "ip_address": {
                    "type": "str",
                    "datarange": string.digits + ".",
                    "len": 15
                }
            }
        }
    }

    sampler = DataSampler()
    sample_dict = sampler.generate(**data_structure)
    print("生成的数据:")
    print(sample_dict)

    # 使用修饰器对叶节点中的 int 和 float 类型进行统计
    result, stats_result = sampler.analyze(sample_dict)
    print("\n统计结果:")
    print(stats_result)