import math
from functools import wraps
from random_nested_sample import random_nested_sample as base_random_nested_sample

def stat_decorator(*stats):
    """
    带参数的修饰器，对返回的样本集所有数值型数据进行统计。
    支持SUM、AVG、VAR、RMSE四项，参数可任意组合。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            samples = func(*args, **kwargs)
            # 提取所有数值型数据
            nums = []
            def extract_numbers(obj):
                if isinstance(obj, (int, float)):
                    nums.append(obj)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        extract_numbers(item)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        extract_numbers(v)
            extract_numbers(samples)
            result = {}
            if not nums:
                for stat in stats:
                    result[stat] = None
                return result
            n = len(nums)
            s = sum(nums)
            mean = s / n
            var = sum((x - mean) ** 2 for x in nums) / n
            rmse = math.sqrt(sum((x - mean) ** 2 for x in nums) / n)
            for stat in stats:
                if stat == 'SUM':
                    result['SUM'] = s
                elif stat == 'AVG':
                    result['AVG'] = mean
                elif stat == 'VAR':
                    result['VAR'] = var
                elif stat == 'RMSE':
                    result['RMSE'] = rmse
            return result
        return wrapper
    return decorator

# 自动填充实验样本结构
sample_struct = {
    "id": int,
    "score": float,
    "info": {
        "age": int,
        "height": float,
        "tags": [str, int]
    },
    "values": [int, float, float]
}

# 用修饰器直接修饰random_nested_sample
@stat_decorator('SUM', 'AVG', 'VAR', 'RMSE')
def random_nested_sample_with_stat(num=5, structure=sample_struct):
    return base_random_nested_sample(num=num, structure=structure)

if __name__ == "__main__":
    stats_result = random_nested_sample_with_stat()
    print("统计结果:")
    for k, v in stats_result.items():
        print(f"{k}: {v}") 
