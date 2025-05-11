import random
import string
from datetime import datetime, timedelta


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

    def generate_user_data(self):
        """
        生成用户数据的示例方法，内部定义了数据结构。

        Returns:
            生成的用户数据字典。
        """
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
        return self.generate(**data_structure)


if __name__ == "__main__":
    sampler = DataSampler()
    sample_dict = sampler.generate_user_data()
    print(sample_dict)