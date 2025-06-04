import random
import string
from typing import Any, Dict, List, Tuple, Union

def generate_samples(**kwargs) -> List[Any]:
    # 获取参数并设置默认值
    n = kwargs.get('n', 1)
    structure = kwargs.get('structure')
    depth_limit = kwargs.get('depth_limit', 5)
    list_length = kwargs.get('list_length', random.randint(3, 10))
    str_length = kwargs.get('str_length', random.randint(5, 15))
    
    if structure is None:
        raise ValueError("必须提供'structure'参数描述数据结构")
    
    def generate_value(data_type, current_depth=0):
        """递归生成单个随机值"""
        if current_depth > depth_limit:
            return None  # 防止无限递归
            
        # 处理基本类型
        if data_type is int:
            return random.randint(1, 1000)
        elif data_type is float:
            return round(random.uniform(0.0, 100.0), 2)
        elif data_type is bool:
            return random.choice([True, False])
        elif data_type is str:
            return ''.join(random.choices(string.ascii_letters + string.digits, k=str_length))
        
        # 处理嵌套结构
        elif isinstance(data_type, list):
            # 列表结构：[元素类型] 或 [具体值]
            if len(data_type) == 1:
                # 统一元素类型的列表
                element_type = data_type[0]
                return [generate_value(element_type, current_depth + 1) 
                        for _ in range(list_length)]
            else:
                # 混合元素类型的列表
                return [generate_value(item, current_depth + 1) 
                        for item in data_type]
        
        elif isinstance(data_type, tuple):
            # 元组结构：(类型1, 类型2, ...)
            return tuple(generate_value(item, current_depth + 1) 
                         for item in data_type)
        
        elif isinstance(data_type, dict):
            # 字典结构：{键: 类型}
            return {
                key: generate_value(value, current_depth + 1)
                for key, value in data_type.items()
            }
        
        # 处理自定义结构或直接值
        else:
            return data_type  # 直接返回非类型说明的值
    
    # 生成n个样本
    return [generate_value(structure) for _ in range(n)]


# 测试示例
if __name__ == "__main__":
    # 示例1：简单的嵌套结构
    simple_samples = generate_samples(
        n=2,
        structure={
            "id": int,
            "name": str,
            "is_active": bool,
            "scores": [float]
        }
    )
    print("简单样本示例:")
    for i, sample in enumerate(simple_samples, 1):
        print(f"样本{i}: {sample}")
    
    # 示例2：复杂的多层嵌套结构
    complex_samples = generate_samples(
        n=3,
        structure={
            "user": {
                "id": int,
                "profile": {
                    "name": str,
                    "age": int,
                    "interests": [str]
                }
            },
            "transactions": ([{
                "id": int,
                "amount": float,
                "items": [(str, int)]
            }],),
            "matrix": [[[int]]]
        },
        depth_limit=6,
        list_length=2
    )
    
    print("\n复杂样本示例:")
    for i, sample in enumerate(complex_samples, 1):
        print(f"样本{i}: {sample}")
