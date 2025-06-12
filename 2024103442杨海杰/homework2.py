import random
import datetime
import string
from collections.abc import Callable
from typing import Any, Dict, List, Tuple


class DataSampler:
    """任意嵌套数据结构随机样本生成器"""

    TYPE_GENERATORS = {
        'int': lambda: random.randint(1, 100),
        'float': lambda: round(random.uniform(0.1, 10.0), 4),
        'str': lambda: ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 8))),
        'bool': lambda: random.choice([True, False]),
        'date': lambda: datetime.datetime.today() - datetime.timedelta(days=random.randint(0, 365 * 2)),
        'datetime': lambda: datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 365 * 2)),
    }

    def generate(self, **kwargs) -> List[Dict]:

        structure = kwargs.get('structure', {})
        sample_count = kwargs.get('sample_count', 1)
        return [self._create_sample(structure) for _ in range(sample_count)]

    def _create_sample(self, spec: Any) -> Any:
        """递归创建单个样本"""
        # 处理可调用生成器
        if callable(spec):
            return self._handle_callable(spec)

        # 处理字典结构
        if isinstance(spec, dict):
            return {key: self._create_sample(value) for key, value in spec.items()}

        # 处理列表结构
        if isinstance(spec, list):
            return [self._create_sample(item) for item in spec]

        # 处理元组结构
        if isinstance(spec, tuple):
            return tuple(self._create_sample(item) for item in spec)

        # 处理预定义类型字符串
        if isinstance(spec, str) and spec in self.TYPE_GENERATORS:
            return self.TYPE_GENERATORS[spec]()

        # 处理基本类型
        if spec in (int, float, str, bool):
            return self._create_basic_type(spec)

        # 直接返回固定值
        return spec

    def _handle_callable(self, spec) -> Any:
        """处理可调用生成器函数"""
        try:
            # 为常见函数提供默认参数
            if spec == random.choice:
                choices = self._random_elements()
                return spec(choices)
            elif spec == random.randint:
                return spec(1, 100)
            elif spec == random.uniform:
                return spec(0.1, 10.0)
            else:
                return spec()
        except Exception:
            return None

    def _create_basic_type(self, spec) -> Any:
        """处理基本数据类型"""
        try:
            if spec == int:
                return self.TYPE_GENERATORS['int']()
            if spec == float:
                return self.TYPE_GENERATORS['float']()
            if spec == str:
                return self.TYPE_GENERATORS['str']()
            if spec == bool:
                return self.TYPE_GENERATORS['bool']()
            return None
        except Exception:
            return None

    def _random_elements(self):
        """生成随机元素集合"""
        return [
            ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 6)))
            for _ in range(random.randint(2, 4))
        ]


# 使用示例
if __name__ == "__main__":
    sampler = DataSampler()

    # 示例1：用户数据结构
    user_structure = {
        "user_id": 'int',
        "username": 'str',
        "email": lambda: f"{random.choice(['user', 'client', 'customer'])}{random.randint(100, 999)}@example.com",
        "is_active": 'bool',
        "signup_date": 'date',
        "account_info": {
            "balance": 'float',
            "last_transaction": lambda: round(random.uniform(-100.0, 100.0), 2),
            "credit_limit": 'int'
        },
        "preferences": [
            {"category": random.choice, "subcategory": lambda: random.choice(['A', 'B', 'C'])},
        ],
        "metadata": {
            "location": lambda: random.choice(['US', 'UK', 'CA', 'AU']),
            "language": 'str',
            "timezone": lambda: random.randint(-12, 12)
        }
    }

    print("用户数据示例:")
    users = sampler.generate(structure=user_structure, sample_count=2)
    for i, user in enumerate(users, 1):
        print(f"\n用户 {i}:")
        # 格式化日期显示
        if 'signup_date' in user:
            user['signup_date'] = user['signup_date'].strftime("%Y-%m-%d")
        for key, value in user.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    # 示例2：产品数据结构
    product_structure = {
        "product_name": 'str',
        "quantity": 'int',
        "weight": 'float',
        "attributes": [
            {"colors": lambda: [''.join(random.choices(string.ascii_uppercase, k=4))
                                for _ in range(2)]},
            {"sizes": lambda: tuple(random.randint(30, 50) for _ in range(3))},
            {"ratings": lambda: [round(random.uniform(1.0, 5.0), 4) for _ in range(5)]},
            {"labels": lambda: tuple(''.join(random.choices(string.ascii_uppercase, k=3))
                                     for _ in range(2))}
        ],
        "category": random.choice,
        "in_stock": 'bool',
        "added_date": 'date'
    }

    print("\n\n产品数据示例:")
    products = sampler.generate(structure=product_structure, sample_count=2)
    for i, product in enumerate(products, 1):
        print(f"\n产品 {i}:")
        # 格式化日期显示
        if 'added_date' in product:
            product['added_date'] = product['added_date'].strftime("%Y-%m-%d")
        for key, value in product.items():
            if key == 'attributes':
                print(f"  {key}:")
                for attr in value:
                    for k, v in attr.items():
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")