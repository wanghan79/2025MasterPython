import random
import string
import uuid
from typing import Any, Dict, Type, Callable, Optional, Union, Tuple

def create_random_object(**kwargs) -> Dict[str, Any]:
    
    result = {}
    
    for field_name, field_spec in kwargs.items():
        # 处理嵌套字典
        if isinstance(field_spec, dict):
            result[field_name] = create_random_object(**field_spec)
            continue
        
        # 处理类型和参数元组
        if isinstance(field_spec, tuple) and len(field_spec) >= 1:
            data_type = field_spec[0]
            params = field_spec[1] if len(field_spec) > 1 else {}
            
            if not isinstance(params, dict):
                params = {}
            
            result[field_name] = generate_random_value(data_type, **params)
            continue
        
        # 处理简单类型
        result[field_name] = generate_random_value(field_spec)
    
    return result

def generate_random_value(data_type: Type, **kwargs) -> Any:
   
    # 整数类型
    if data_type == int:
        min_val = kwargs.get('min', -1000000)
        max_val = kwargs.get('max', 1000000)
        return random.randint(min_val, max_val)
    
    # 浮点数类型
    elif data_type == float:
        min_val = kwargs.get('min', -1000.0)
        max_val = kwargs.get('max', 1000.0)
        return min_val + random.random() * (max_val - min_val)
    
    # 字符串类型
    elif data_type == str:
        length = kwargs.get('length', 10)
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    
    # 布尔类型
    elif data_type == bool:
        return random.choice([True, False])
    
    # UUID类型
    elif data_type == uuid.UUID:
        return uuid.uuid4()
    
    # 尝试使用类型构造函数
    else:
        try:
            return data_type()
        except Exception:
            return None

# 示例使用
if __name__ == "__main__":
    # 生成简单对象
    simple_obj = create_random_object(
        id=int,
        name=str,
        score=(float, {'min': 0, 'max': 100}),
        is_active=bool
    )
    print("简单对象:", simple_obj)
    
    # 生成嵌套对象
    nested_obj = create_random_object(
        user_id=int,
        username=(str, {'length': 8}),
        profile=dict(
            age=(int, {'min': 18, 'max': 65}),
            email=str,
            preferences=dict(
                theme=str,
                notifications=bool
            )
        )
    )
    print("嵌套对象:", nested_obj)