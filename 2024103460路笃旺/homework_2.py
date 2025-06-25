
import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, Union, List, Tuple

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_date(start_year=2000, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')

def generate_value(dtype: Union[str, list, tuple, dict]) -> Any:
    """
    根据数据类型递归生成随机值。
    """
    if isinstance(dtype, str):
        if dtype == 'int':
            return random.randint(0, 10000)
        elif dtype == 'float':
            return round(random.uniform(0, 10000), 2)
        elif dtype == 'str':
            return random_string()
        elif dtype == 'bool':
            return random.choice([True, False])
        elif dtype == 'date':
            return random_date()
        else:
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
