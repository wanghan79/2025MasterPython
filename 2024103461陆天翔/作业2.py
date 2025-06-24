import random as rd
import string as st
from datetime import date as dt_date
from typing import Any, Dict, List, Optional, Tuple


class MockDataGenerator:
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed:
            rd.seed(random_seed)

        # 数据类型与生成方法的映射
        self.data_creators = {
            int: self._create_integer,
            float: self._create_float_num,
            str: self._create_random_str,
            list: self._create_random_list,
            tuple: self._create_fixed_sequence,
            dict: self._create_key_value_pairs,
            bool: self._create_boolean_val,
            dt_date: self._create_random_date
        }

    def make_multiple_instances(self, quantity: int, **settings) -> List[Any]:
        """批量生成数据样本"""
        return [self._make_single_instance(settings) for _ in range(quantity)]

    def _make_single_instance(self, params: Dict) -> Any:
        """生成单个数据样本"""
        required_type = params.get('type')

        if required_type not in self.data_creators:
            raise TypeError(f"不支持的数据类型: {required_type}")

        return self.data_creators[required_type](params)

    def _create_integer(self, params: Dict) -> int:
        """产生随机整数"""
        lower = params.get('min', 0)
        upper = params.get('max', 100)
        return rd.randint(lower, upper)

    def _create_float_num(self, params: Dict) -> float:
        """产生随机浮点数"""
        lower = params.get('min', 0.0)
        upper = params.get('max', 1.0)
        decimal_places = params.get('precision', 2)
        return round(rd.uniform(lower, upper), decimal_places)

    def _create_random_str(self, params: Dict) -> str:
        """生成随机字符串"""
        str_len = params.get('length', 10)
        char_set = params.get('charset', st.ascii_letters + st.digits)
        return ''.join(rd.choices(char_set, k=str_len))

    def _create_random_list(self, params: Dict) -> list:
        """生成可变长度列表"""
        min_len = params.get('min_length', 1)
        max_len = params.get('max_length', 10)
        actual_len = rd.randint(min_len, max_len)

        item_config = params.get('element_config', {})
        return [self._make_single_instance(item_config) for _ in range(actual_len)]

    def _create_fixed_sequence(self, params: Dict) -> tuple:
        """生成固定长度元组"""
        items_config = params.get('elements_config', [{}])
        return tuple(self._make_single_instance(cfg) for cfg in items_config)

    def _create_key_value_pairs(self, params: Dict) -> dict:
        """生成字典数据"""
        field_definitions = params.get('fields', {})
        return {k: self._make_single_instance(v) for k, v in field_definitions.items()}

    def _create_boolean_val(self, params: Dict) -> bool:
        """生成布尔值"""
        true_chance = params.get('true_prob', 0.5)
        return rd.random() < true_chance

    def _create_random_date(self, params: Dict) -> dt_date:
        """生成随机日期"""
        start = params.get('start_date', dt_date(2000, 1, 1))
        end = params.get('end_date', dt_date.today())
        delta_days = (end - start).days
        return start + dt_date.resolution * rd.randint(0, delta_days)


# 使用示例
if __name__ == "__main__":
    # 初始化数据生成器
    data_factory = MockDataGenerator(random_seed=42)

    # 定义用户数据结构模板
    user_template = {
        'type': dict,
        'fields': {
            'user_id': {'type': int, 'min': 1, 'max': 10000},
            'username': {'type': str, 'length': 8},
            'age': {'type': int, 'min': 18, 'max': 99},
            'active_status': {'type': bool, 'true_prob': 0.8},
            'height': {'type': float, 'min': 1.5, 'max': 2.0, 'precision': 2},
            'join_date': {'type': dt_date, 'start_date': dt_date(2020, 1, 1)},
            'interests': {
                'type': list,
                'min_length': 2,
                'max_length': 5,
                'element_config': {'type': str, 'length': 6}
            },
            'contact': {
                'type': dict,
                'fields': {
                    'street': {'type': str, 'length': 15},
                    'city': {'type': str, 'length': 10},
                    'postcode': {'type': str, 'length': 5, 'charset': st.digits}
                }
            }
        }
    }

    # 生成10个用户数据
    generated_users = data_factory.make_multiple_instances(10, **user_template)

    # 输出结果
    for idx, user_data in enumerate(generated_users, 1):
        print(f"生成用户 #{idx}:")
        for field, value in user_data.items():
            print(f"  {field}: {value}")
        print()