import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, Union, List, Tuple

# 1. 预计算常用字符串集合，避免重复拼接
CHARACTERS = string.ascii_letters + string.digits

def random_string(length=8):
    """使用预计算的字符集生成随机字符串"""
    return ''.join(random.choices(CHARACTERS, k=length))

def random_date(start_year=2000, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')

# 3. 使用类型处理器映射替代if-else链
TYPE_HANDLERS = {
    'int': lambda: random.randint(0, 10000),
    'float': lambda: round(random.uniform(0, 10000), 2),
    'str': random_string,
    'bool': lambda: random.choice([True, False]),
    'date': random_date
}

def generate_value(dtype: Union[str, list, tuple, dict]) -> Any:
    """
    根据数据类型递归生成随机值。
    """
    if isinstance(dtype, str):
        # 使用类型处理器映射
        handler = TYPE_HANDLERS.get(dtype)
        if handler:
            return handler()
        raise ValueError(f"Unsupported type: {dtype}")

    elif isinstance(dtype, list):  # 代表 list 中嵌套的结构
        if not dtype:
            return []
        element_type = dtype[0]
        return [generate_value(element_type) for _ in range(random.randint(1, 3))]

    elif isinstance(dtype, tuple):
        return tuple(generate_value(t) for t in dtype)

    elif isinstance(dtype, dict):
        return {k: generate_value(v) for k, v in dtype.items()}

    else:
        raise TypeError(f"Invalid type: {type(dtype)}")


def DataSampler(sample_structure: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
    """
        根据指定结构生成嵌套样本数据。
        参数:
            sample_structure: dict，描述数据字段及类型
            kwargs:
                count: int，样本数量（默认 1）
        返回:
            List[Dict]，样本数据集合
        """
    count = kwargs.get('count', 1)
    return [generate_value(sample_structure) for _ in range(count)]

if __name__ == "__main__":
    structure = {
        'user_id': 'int',
        'name': 'str',
        'is_active': 'bool',
        'signup_date': 'date',
        'score': 'float',
        'tags': ['str'],
        'history': [
            {'item_id': 'int', 'timestamp': 'date'}
        ],
        'location': ('float', 'float')
    }

    samples = DataSampler(structure, count=5)
    for i, sample in enumerate(samples, 1):
        print(f"样本 {i}：{sample}")
