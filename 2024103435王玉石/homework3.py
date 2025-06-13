import random
import numpy as np

# StaticRes 装饰器
def StaticRes(*stat_types):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 调用 dataSampling 函数，生成数据
            result = func(*args, **kwargs)
            # 将结果转换为列表以进行统计计算
            result_list = list(result)
            # 打印生成的数据
            print("生成的数据:", result_list)
            # 计算请求的统计指标
            stats = {}
            if len(result_list) == 0:
                return stats  # 如果结果为空，直接返回空统计信息
            # 计算SUM
            if "SUM" in stat_types:
                stats["SUM"] = sum(result_list)
            # 计算AVG（平均值）
            if "AVG" in stat_types:
                stats["AVG"] = np.mean(result_list)
            # 计算VAR（方差）
            if "VAR" in stat_types:
                stats["VAR"] = np.var(result_list)
            # 计算RMSE（均方根误差）
            if "RMSE" in stat_types:
                rmse = np.sqrt(np.mean(np.square(np.array(result_list) - np.mean(result_list))))
                stats["RMSE"] = rmse
            return stats

        return wrapper

    return decorator


# 数据采样函数，使用 **kwargs
@StaticRes("SUM", "AVG", "VAR", "RMSE")
def dataSampling(**kwargs):
    datatype = kwargs.get("datatype")
    datarange = kwargs.get("datarange")
    num = kwargs.get("num")
    strlen = kwargs.get("strlen", 8)

    result = set()
    for index in range(0, num):
        if datatype is int:
            it = iter(datarange)
            item = random.randint(next(it), next(it))
            result.add(item)
        elif datatype is float:
            it = iter(datarange)
            item = random.uniform(next(it), next(it))
            result.add(item)
        elif datatype is str:
            item = ''.join(random.SystemRandom().choice(datarange) for _ in range(strlen))
            result.add(item)
        else:
            continue
    return result


# 示例调用
kwargs = {
    "datatype": int,
    "datarange": (1, 100),
    "num": 10
}
result = dataSampling(**kwargs)
print("统计结果:", result)
