import random
import string

def create_random_object(**kwargs):
    """
    根据 kwargs 生成随机对象，支持递归生成嵌套对象。

    参数:
        kwargs: 键为属性名，值为生成随机值的规则：
            - int: 生成一个随机整数（64 位随机整数，正负随机）
            - float: 生成一个随机浮点数（随机指数生成，不限制范围）
            - str: 生成一个长度为 10 的随机字符串
            - dict: 递归生成嵌套对象
            - 其他类型: 默认调用 rule() 初始化对象

    返回:
        一个字典，包含生成的随机属性和值。
    """
    obj = {}
    for key, rule in kwargs.items():
        if rule == int:
            # 生成 64 位随机整数，并随机确定正负
            num = random.getrandbits(64)
            obj[key] = num if random.choice([True, False]) else -num
        elif rule == float:
            # 生成随机浮点数：随机选取指数与基数，不限制数值范围
            exponent = random.randint(-10, 10)
            base = random.random()
            num = base * (10 ** exponent)
            obj[key] = num if random.choice([True, False]) else -num
        elif rule == str:
            # 生成长度为 10 的随机字符串，由字母和数字组成
            obj[key] = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        elif rule == bool:
            obj[key] = random.choice([True, False])
        elif isinstance(rule, dict):
            # 递归生成嵌套对象
            obj[key] = create_random_object(**rule)
        else:
            try:
                # 默认初始化对象
                obj[key] = rule()
            except Exception as e:
                # 若初始化失败，则返回 None
                obj[key] = None
    return obj

# 示例使用
if __name__ == "__main__":
    # 假设 MyClass 是一个用户自定义的类
    class MyClass:
        def __init__(self):
            self.value = "默认值"
        def __repr__(self):
            return f"MyClass(value={self.value})"

    random_obj = create_random_object(
        age=int,        # 随机生成一个 64 位整数
        score=float,    # 随机生成一个浮点数
        name=str,       # 随机生成一个长度为 10 的字符串
        custom=MyClass, # 默认初始化 MyClass 对象
        boolValue=bool,
        nested=dict(    # 递归生成嵌套对象
            nested_age=int,
            nested_name=str
        )
    )
    print(random_obj)
