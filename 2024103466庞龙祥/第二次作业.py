import random
import string
import time
from datetime import date, timedelta, datetime
from typing import Any, Callable, Dict, List, Tuple, Union, Optional


class DataSampler:
    """高级随机数据生成器 - 创建任意嵌套结构的随机样本"""

    def __init__(self, custom_types: Dict[str, Callable] = None):
        # 默认类型生成器
        self.type_generators = {
            int: self._generate_int,
            float: self._generate_float,
            str: self._generate_str,
            bool: self._generate_bool,
            date: self._generate_date,
            list: self._generate_list,
            tuple: self._generate_tuple,
            dict: self._generate_dict
        }

        # 字符串简写映射
        self.type_aliases = {
            'int': int,
            'float': float,
            'str': str,
            'string': str,
            'bool': bool,
            'boolean': bool,
            'date': date,
            'list': list,
            'tuple': tuple,
            'dict': dict,
            'dictionary': dict
        }

        # 添加自定义类型生成器
        if custom_types:
            for type_name, generator in custom_types.items():
                self.add_custom_type(type_name, generator)

    def add_custom_type(self, type_name: str, generator: Callable):
        """添加自定义数据类型生成器"""
        self.type_aliases[type_name] = generator

    def _resolve_type(self, schema: Any) -> Any:
        """解析模式类型，支持类型简写和自定义生成器"""
        # 处理字符串类型别名
        if isinstance(schema, str) and schema in self.type_aliases:
            return self.type_aliases[schema]

        # 处理类型简写
        if type(schema) == type and schema in self.type_generators:
            return schema

        # 处理固定值生成器
        if callable(schema):
            return schema

        return schema

    def _generate_int(self, min_val: int = 0, max_val: int = 100) -> int:
        """生成随机整数"""
        return random.randint(min_val, max_val)

    def _generate_float(self, min_val: float = 0.0, max_val: float = 100.0,
                        decimals: int = 2) -> float:
        """生成随机浮点数"""
        value = random.uniform(min_val, max_val)
        return round(value, decimals)

    def _generate_str(self, min_len: int = 5, max_len: int = 15,
                      prefix: str = "", suffix: str = "") -> str:
        """生成随机字符串"""
        length = random.randint(min_len, max_len)
        chars = string.ascii_letters + string.digits
        random_chars = ''.join(random.choices(chars, k=length))
        return f"{prefix}{random_chars}{suffix}"

    def _generate_bool(self) -> bool:
        """生成随机布尔值"""
        return random.choice([True, False])

    def _generate_date(self, start_date: date = date(2000, 1, 1),
                       end_date: date = date.today()) -> date:
        """生成随机日期"""
        days_diff = (end_date - start_date).days
        random_days = random.randint(0, days_diff)
        return start_date + timedelta(days=random_days)

    def _generate_list(self, schema: Any,
                       min_len: int = 1, max_len: int = 5) -> list:
        """生成随机列表"""
        length = random.randint(min_len, max_len)
        return [self.generate_sample(schema) for _ in range(length)]

    def _generate_tuple(self, schema: Tuple) -> tuple:
        """生成元组"""
        return tuple(self.generate_sample(item) for item in schema)

    def _generate_dict(self, schema: Dict[str, Any]) -> dict:
        """生成字典，处理所有可能的类型"""
        result = {}
        for key, value in schema.items():
            try:
                result[key] = self.generate_sample(value)
            except Exception as e:
                raise ValueError(f"生成字段 '{key}' 时出错: {str(e)}") from e
        return result

    def generate_sample(self, schema: Any, **kwargs) -> Any:
        """
        根据模式生成单个样本

        参数:
        schema -- 数据模式定义
        **kwargs -- 类型特定的额外参数

        返回:
        随机生成的样本
        """
        # 处理固定值模式（int, float, str, bool等）
        if isinstance(schema, (int, float, str, bool, date)):
            return schema

        # 处理类型简写和自定义生成器
        resolved_schema = self._resolve_type(schema)

        # 处理基本类型生成
        if resolved_schema == int:
            return self._generate_int(**kwargs)
        elif resolved_schema == float:
            return self._generate_float(**kwargs)
        elif resolved_schema == str:
            return self._generate_str(**kwargs)
        elif resolved_schema == bool:
            return self._generate_bool(**kwargs)
        elif resolved_schema == date:
            return self._generate_date(**kwargs)

        # 处理列表类型
        elif isinstance(resolved_schema, list):
            if len(resolved_schema) == 1:
                return self._generate_list(resolved_schema[0], **kwargs)
            return [self.generate_sample(item, **kwargs) for item in resolved_schema]

        # 处理元组类型
        elif isinstance(resolved_schema, tuple):
            return self._generate_tuple(resolved_schema)

        # 处理字典类型
        elif isinstance(resolved_schema, dict):
            return self._generate_dict(resolved_schema)

        # 处理可调用对象（自定义生成器）
        elif callable(resolved_schema):
            return resolved_schema(**kwargs)

        else:
            raise ValueError(f"不支持的模式类型: {type(schema)}")

    def generate_samples(self, n: int = 1, **schemas) -> List[dict]:
        """
        生成多个样本

        参数:
        n -- 要生成的样本数量
        schemas -- 字段模式定义

        返回:
        包含n个样本的列表
        """
        samples = []
        for _ in range(n):
            sample = {}
            for field, schema in schemas.items():
                try:
                    # 处理包含元数据的复杂模式
                    if isinstance(schema, dict) and 'schema' in schema:
                        field_kwargs = {k: v for k, v in schema.items() if k != 'schema'}
                        sample[field] = self.generate_sample(schema['schema'], **field_kwargs)
                    else:
                        sample[field] = self.generate_sample(schema)
                except Exception as e:
                    raise ValueError(f"生成字段 '{field}' 时出错: {str(e)}") from e
            samples.append(sample)
        return samples

    # 预设数据模式
    def user_sample(self) -> dict:
        """用户样本模式"""
        return {
            'id': 'int',
            'name': str,
            'email': self._generate_email,
            'age': {'schema': int, 'min_val': 18, 'max_val': 90},
            'is_active': bool,
            'created_at': date,
            'preferences': {
                'theme': 'str',
                'notifications': bool
            }
        }

    def product_sample(self) -> dict:
        """产品样本模式"""
        return {
            'id': 'int',
            'name': str,
            'description': {'schema': str, 'min_len': 20, 'max_len': 100},
            'price': float,
            'in_stock': bool,
            'categories': {'schema': ['electronics', 'books', 'clothing', 'home'], 'min_len': 1, 'max_len': 3}
        }

    def order_sample(self, max_items: int = 5) -> dict:
        """订单样本模式"""
        return {
            'id': 'int',
            'user_id': 'int',
            'items': {
                'schema': {
                    'product_id': int,
                    'name': str,
                    'quantity': {'schema': int, 'min_val': 1, 'max_val': 10},
                    'price': float
                },
                'min_len': 1,
                'max_len': max_items
            },
            'total': float,
            'created_at': date,
            'status': {
                'schema': lambda: random.choice(['pending', 'processing', 'shipped', 'delivered'])
            }
        }

    # 辅助方法
    def _generate_email(self, domains: List[str] = None) -> str:
        """生成随机电子邮件地址"""
        domains = domains or ["gmail.com", "yahoo.com", "hotmail.com", "example.com"]
        username = self._generate_str(min_len=5, max_len=10)
        domain = random.choice(domains)
        return f"{username}@{domain}"


if __name__ == "__main__":
    # 创建数据采样器实例
    sampler = DataSampler()

    # 测试1: 生成用户数据
    print("=== 测试用户数据生成 ===")
    user_schema = {
        "id": int,
        "name": str,
        "email": "str",
        "age": {"schema": int, "min_val": 18, "max_val": 65},
        "is_admin": bool,
        "created_at": date,
        "preferences": {
            "theme": ["light", "dark"],
            "notifications": bool
        }
    }

    try:
        users = sampler.generate_samples(n=3, **user_schema)
        for i, user in enumerate(users):
            print(f"用户 {i + 1}: {user}")
        print("用户数据生成测试成功！\n")
    except Exception as e:
        print(f"用户数据生成错误: {str(e)}")

    # 测试2: 生成产品数据
    print("=== 测试产品数据生成 ===")
    try:
        products = sampler.generate_samples(n=2, **sampler.product_sample())
        for i, product in enumerate(products):
            print(f"产品 {i + 1}: {product['name']} (${product['price']})")
            print(f"  描述: {product['description'][:50]}...")
            print(f"  分类: {product['categories']}\n")
        print("产品数据生成测试成功！\n")
    except Exception as e:
        print(f"产品数据生成错误: {str(e)}")

    # 测试3: 生成订单数据
    print("=== 测试订单数据生成 ===")
    try:
        orders = sampler.generate_samples(n=2, **sampler.order_sample(max_items=3))
        for i, order in enumerate(orders):
            print(f"订单 {i + 1} (总计: ${order['total']:.2f}, 状态: {order['status']})")
            for item in order['items']:
                print(f"  - {item['name']}: ${item['price']} x {item['quantity']}")
            print()
        print("订单数据生成测试成功！\n")
    except Exception as e:
        print(f"订单数据生成错误: {str(e)}")

    # 测试4: 复杂嵌套结构
    print("=== 测试复杂嵌套结构 ===")
    company_schema = {
        "name": str,
        "founded": date,
        "departments": {
            "min_len": 2,
            "max_len": 4,
            "schema": {
                "name": str,
                "manager": {
                    "name": str,
                    "title": {
                        "schema": lambda: random.choice(["Director", "VP", "Manager"])
                    }
                },
                "employees": {
                    "min_len": 3,
                    "max_len": 5,
                    "schema": {
                        "id": int,
                        "name": str,
                        "position": str,
                        "salary": {"schema": float, "min_val": 30000, "max_val": 150000}
                    }
                }
            }
        }
    }

    try:
        companies = sampler.generate_samples(n=1, **company_schema)
        company = companies[0]
        print(f"公司: {company['name']} (创立于 {company['founded']})")

        for dept in company["departments"]:
            print(f"  - 部门: {dept['name']}, 主管: {dept['manager']['name']} ({dept['manager']['title']})")
            print("    员工:")
            for emp in dept["employees"]:
                print(f"      • {emp['name']} ({emp['position']}): ${emp['salary']:,.2f}")
        print("复杂嵌套结构生成测试成功！")
    except Exception as e:
        print(f"复杂嵌套结构生成错误: {str(e)}")