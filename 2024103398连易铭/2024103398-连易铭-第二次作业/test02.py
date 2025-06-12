import random
import string

# 一个函数，用来生成一份随机的“数据字典”
def my_data_generator(**rules):
    result = {}

    for name in rules:
        rule = rules[name]

        # 如果是 int 类型
        if rule == int:
            result[name] = random.randint(0, 100)

        # 如果是 float 类型
        elif rule == float:
            result[name] = random.uniform(0, 100)

        # 如果是 str 类型
        elif rule == str:
            letters = string.ascii_letters
            s = ""
            for i in range(10):
                s += random.choice(letters)
            result[name] = s

        # 如果是 bool 类型
        elif rule == bool:
            result[name] = random.choice([True, False])

        # 如果是两个数的范围，比如 (0, 100)
        elif type(rule) == tuple or type(rule) == list:
            if len(rule) == 2:
                if type(rule[0]) == int and type(rule[1]) == int:
                    result[name] = random.randint(rule[0], rule[1])
                elif type(rule[0]) == float and type(rule[1]) == float:
                    result[name] = random.uniform(rule[0], rule[1])
                else:
                    result[name] = random.choice(rule)
            else:
                result[name] = random.choice(rule)

        # 其他情况就直接当成是固定的值
        else:
            result[name] = rule

    return result


# 稍微写点示例
if __name__ == "__main__":
    data = my_data_generator(
        age=int,
        height=float,
        name=str,
        is_student=bool,
        test_score=(0, 100),
        preferred_colors=("red", "blue", "green")
    )

    print("Here is your random data:")
    for key in data:
        print(key + " =>", data[key])
