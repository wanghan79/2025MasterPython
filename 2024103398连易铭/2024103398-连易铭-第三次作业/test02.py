import random
import string
import numpy as np

# 假装我们自己写了这个函数，其实从别的文件里来的
def my_create_object(**configs):
    result = {}
    for key in configs:
        rule = configs[key]
        if rule == int:
            result[key] = random.getrandbits(64) * random.choice([-1, 1])
        elif rule == float:
            base = random.random()
            power = random.randint(-308, 308)
            result[key] = base * (10 ** power) * random.choice([-1, 1])
        elif rule == str:
            result[key] = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        elif rule == bool:
            result[key] = random.choice([True, False])
        elif isinstance(rule, dict):
            result[key] = my_create_object(**rule)
        else:
            try:
                result[key] = rule()
            except:
                result[key] = None
    return result


# 生成好多份随机数据
def make_samples(how_many=1, **configs):
    data = []
    for i in range(how_many):
        one = my_create_object(**configs)
        data.append(one)
    return data


# 把对象里的数字提取出来，记录路径和值
def take_all_numbers(obj, prefix=''):
    out = {}

    if isinstance(obj, (int, float)):
        path = prefix if prefix else "value"
        out[path] = [float(obj)]
    elif isinstance(obj, dict):
        for k in obj:
            new_key = f"{prefix}.{k}" if prefix else k
            sub = take_all_numbers(obj[k], new_key)
            out.update(sub)
    elif isinstance(obj, (list, tuple)):
        for i in range(len(obj)):
            new_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            sub = take_all_numbers(obj[i], new_key)
            out.update(sub)

    return out


# 用来加统计功能的装饰器函数
def my_stats(stat_kind='mean'):

    def inner_decorator(func):
        def wrapped(*args, **kwargs):
            all_data = func(*args, **kwargs)

            if not isinstance(all_data, (list, tuple)):
                all_data = [all_data]

            numbers_dict = {}

            # 收集所有数值
            for item in all_data:
                one_result = take_all_numbers(item)
                for field in one_result:
                    if field not in numbers_dict:
                        numbers_dict[field] = []
                    numbers_dict[field].extend(one_result[field])

            # 开始做统计
            stats = {}
            for key in numbers_dict:
                vals = numbers_dict[key]
                if not vals:
                    stats[key] = None
                elif stat_kind == 'mean':
                    stats[key] = float(np.mean(vals))
                elif stat_kind == 'variance':
                    stats[key] = float(np.var(vals))
                elif stat_kind == 'rmse':
                    stats[key] = float(np.sqrt(np.mean(np.square(vals))))
                elif stat_kind == 'sum':
                    stats[key] = float(np.sum(vals))
                else:
                    stats[key] = None

            return stats

        return wrapped
    return inner_decorator


# ======= 测试函数 =======

@my_stats(stat_kind='mean')
def test_mean(n=5):
    return make_samples(n, value=int)


@my_stats(stat_kind='variance')
def test_var(n=5):
    return make_samples(n, value=int)


@my_stats(stat_kind='rmse')
def test_rmse(n=5):
    return make_samples(n, value=int)


@my_stats(stat_kind='sum')
def test_sum(n=5):
    return make_samples(n, value=int)


@my_stats(stat_kind='mean')
def test_complex(n=5):
    return make_samples(
        n,
        age=int,
        score=float,
        name=str,
        nested={
            "nested_age": int,
            "nested_name": str
        }
    )


# 输出结果看一下
if __name__ == "__main__":
    print("平均值：", test_mean(5))
    print("方差：", test_var(5))
    print("RMSE：", test_rmse(5))
    print("总和：", test_sum(5))
    print("\n复杂对象的平均：", test_complex(3))
