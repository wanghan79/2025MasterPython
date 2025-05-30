import random
import string

# 这个函数可以生成一个包含随机值的字典
def my_random_maker(**things):
    result = {}

    # 遍历所有传进来的参数
    for key in things:
        value = things[key]

        # 如果是整数类型
        if value == int:
            result[key] = random.randint(0, 100)
        # 如果是浮点数类型
        elif value == float:
            result[key] = random.uniform(0, 100)
        # 如果是字符串类型
        elif value == str:
            letters = string.ascii_letters
            random_string = ""
            for i in range(10):
                random_string += random.choice(letters)
            result[key] = random_string
        # 如果是布尔值
        elif value == bool:
            result[key] = random.choice([True, False])
        # 如果是两个数的范围或者是一个选项组
        elif type(value) == tuple or type(value) == list:
            if len(value) == 2:
                # 如果是两个整数
                if type(value[0]) == int and type(value[1]) == int:
                    result[key] = random.randint(value[0], value[1])
                # 如果是两个小数
                elif type(value[0]) == float and type(value[1]) == float:
                    result[key] = random.uniform(value[0], value[1])
                # 其他情况，像字符串列表
                else:
                    result[key] = random.choice(value)
            else:
                result[key] = random.choice(value)
        # 其他就直接返回
        else:
            result[key] = value

    return result

# 用上面这个函数来生成一个字典
my_object = my_random_maker(
    age=int,
    height=float,
    name=str,
    is_student=bool,
    score=(0, 100),
    favorite_colors=("red", "blue", "green")
)

# 打印出来
print(my_object)
