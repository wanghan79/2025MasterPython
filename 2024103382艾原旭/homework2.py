import random
import string
from datetime import datetime, timedelta
from typing import Any, Union, List, Dict, Tuple


class DataSampler:
    """Random random data generator"""

    @staticmethod
    def _random_int(**kwargs) -> int:
        min_val = kwargs.get('min', 0)
        max_val = kwargs.get('max', 100)
        return random.randint(min_val, max_val)

    @staticmethod
    def _random_float(**kwargs) -> float:
        min_val = kwargs.get('min', 0)
        max_val = kwargs.get('max', 100)
        precision = kwargs.get('precision', 2)
        return round(random.uniform(min_val, max_val), precision)

    @staticmethod
    def _random_str(**kwargs) -> str:
        length = kwargs.get('length', 8)
        chars = kwargs.get('chars', string.ascii_letters + string.digits)
        return ''.join(random.choice(chars) for _ in range(length))

    @staticmethod
    def _random_bool(**kwargs) -> bool:
        return random.choice([True, False])

    @staticmethod
    def _random_date(**kwargs) -> str:
        start = kwargs.get('start', datetime(2000, 1, 1))
        end = kwargs.get('end', datetime(2023, 12, 31))
        delta = end - start
        random_days = random.randint(0, delta.days)
        return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')

    @staticmethod
    def _random_any(**kwargs) -> Any:
        """process "Any" type,and randomly choose basic type"""
        types = [int, float, str, bool, datetime]
        return DataSampler._generate_value(random.choice(types), **kwargs)

    @staticmethod
    def _random_list(spec: Any, size: int, **kwargs) -> List[Any]:
        return [DataSampler._generate_value(spec, **kwargs) for _ in range(size)]

    @staticmethod
    def _random_tuple(spec: Any, size: int, **kwargs) -> Tuple[Any]:
        return tuple(DataSampler._random_list(spec, size, **kwargs))

    @staticmethod
    def _random_dict(spec: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return {key: DataSampler._generate_value(value, **kwargs) for key, value in spec.items()}

    @staticmethod
    def _generate_value(spec: Any, **kwargs) -> Any:
        """generate random value by type"""
        if spec is Any:
            return DataSampler._random_any(**kwargs)
        elif isinstance(spec, type):
            if spec == int:
                return DataSampler._random_int(**kwargs)
            elif spec == float:
                return DataSampler._random_float(**kwargs)
            elif spec == str:
                return DataSampler._random_str(**kwargs)
            elif spec == bool:
                return DataSampler._random_bool(**kwargs)
            elif spec == datetime:
                return DataSampler._random_date(**kwargs)
            else:
                raise ValueError(f"不支持的类型: {spec}")
        elif isinstance(spec, dict):
            return DataSampler._random_dict(spec, **kwargs)
        elif isinstance(spec, (list, tuple)):
            container_type = list if isinstance(spec, list) else tuple
            if len(spec) != 1:
                raise ValueError("列表/元组规范应只包含一个元素，表示元素类型")
            size = kwargs.get('size', 3)  # 默认容器大小
            return DataSampler._random_list(spec[0], size, **kwargs) if container_type is list \
                else DataSampler._random_tuple(spec[0], size, **kwargs)
        else:
            raise ValueError(f"无效的类型规范: {spec}")

    @staticmethod
    def generate_samples(spec: Any, num_samples: int = 1, **kwargs) -> Union[List[Any], Any]:
        """
        generate random samples

        input:
            spec: 数据结构规范，可以是类型或嵌套结构
            num_samples: 样本数量
            kwargs: 各类型的生成参数

        output:
            当num_samples=1时返回单个样本，否则返回样本列表
        """
        if num_samples < 1:
            raise ValueError("样本数量必须大于0")

        samples = [DataSampler._generate_value(spec, **kwargs) for _ in range(num_samples)]
        return samples[0] if num_samples == 1 else samples


if __name__ == "__main__":
    user_spec = {
        "id": int,
        "name": str,
        "is_active": bool,
        "signup_date": datetime,
        "balance": float,
        "contacts": [str],
        "preferences": {
            "theme": str,
            "notifications": bool
        },
        "metadata": (Any,)
    }

    sampler = DataSampler()

    single_user = sampler.generate_samples(user_spec)
    print("单个用户样本:")
    print(single_user)

    multiple_users = sampler.generate_samples(user_spec, num_samples=3)
    print("\n多个用户样本:")
    for i, user in enumerate(multiple_users, 1):
        print(f"用户{i}: {user}")

    custom_user = sampler.generate_samples(
        user_spec,
        str__length=10,
        int__min=1000,
        int__max=9999,
        contacts__size=2
    )
    print("\n自定义参数生成的用户:")
    print(custom_user)