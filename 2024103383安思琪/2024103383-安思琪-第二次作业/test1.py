import random
import string


def generate_random_data(**type_configs):

    result = {}

    for param_name, type_rule in type_configs.items():
        # 处理整数类型
        if type_rule is int:
            result[param_name] = random.randint(0, 100)

        # 处理浮点数类型
        elif type_rule is float:
            result[param_name] = random.uniform(0, 100)

        # 处理字符串类型
        elif type_rule is str:
            rand_str = ''.join(random.choices(string.ascii_letters, k=10))
            result[param_name] = rand_str

        # 处理布尔类型
        elif type_rule is bool:
            result[param_name] = random.choice([True, False])

        # 处理数值范围配置
        elif isinstance(type_rule, (tuple, list)) and len(type_rule) == 2:
            # 整数范围
            if all(isinstance(x, int) for x in type_rule):
                min_val, max_val = type_rule
                result[param_name] = random.randint(min_val, max_val)
            # 浮点数范围
            elif all(isinstance(x, float) for x in type_rule):
                min_val, max_val = type_rule
                result[param_name] = random.uniform(min_val, max_val)
            # 普通元组选择
            else:
                result[param_name] = random.choice(type_rule)

        # 处理固定值或选项列表
        else:
            result[param_name] = type_rule

    return result


# 使用示例
if __name__ == "__main__":
    sample_data = generate_random_data(
        age=int,
        height=float,
        name=str,
        is_student=bool,
        test_score=(0, 100),
        preferred_colors=("red", "blue", "green")
    )

    print("Generated Random Data:")
    for k, v in sample_data.items():
        print(f"{k:12} => {v}")