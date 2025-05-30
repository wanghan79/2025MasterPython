import random
import string

# 自己写的生成器
def my_random_object(**configs):
    result = {}

    for key in configs:
        rule = configs[key]

        # 如果是整数类型
        if rule == int:
            num = random.getrandbits(64)
            if random.choice([True, False]):
                result[key] = num
            else:
                result[key] = -num

        # 如果是浮点数类型
        elif rule == float:
            base = random.random()
            exp = random.randint(-308, 308)
            num = base * (10 ** exp)
            if random.choice([True, False]):
                result[key] = num
            else:
                result[key] = -num

        # 字符串类型
        elif rule == str:
            s = ""
            for i in range(10):
                s += random.choice(string.ascii_letters + string.digits)
            result[key] = s

        # 布尔值类型
        elif rule == bool:
            result[key] = random.choice([True, False])

        # 如果是嵌套的字典
        elif type(rule) == dict:
            result[key] = my_random_object(**rule)

        # 如果是类，就尝试初始化
        else:
            try:
                result[key] = rule()
            except:
                result[key] = None

    return result

# 举个例子
if __name__ == "__main__":
    class MyClass:
        def __init__(self):
            self.value = "默认值"
        def __repr__(self):
            return "MyClass(value=" + self.value + ")"

    data = my_random_object(
        age=int,
        score=float,
        name=str,
        custom=MyClass,
        boolValue=bool,
        nested={
            "nested_age": int,
            "nested_name": str
        }
    )

    print(data)
