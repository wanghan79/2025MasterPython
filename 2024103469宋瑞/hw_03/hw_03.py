"""
数据挖掘-王晗 第三次作业 2025年5月6日
函数功能：
    1、写一个修饰器，实现 SUM、AVG、方差、RMSE，用于前述 data_sampler
"""
import random
import string
import math
import pprint
import datetime

def data_sampler(num, **kwargs):
    result = []
    for _ in range(num):
        element = []
        for key, value in kwargs.items():
            if key == "int":
                tmp = random.randint(value['data_range'][0], value['data_range'][1])
            elif key == "float":
                tmp = round(random.uniform(value['data_range'][0], value['data_range'][1]), value.get('decimal_places', 2))
            elif key == "str":
                tmp = ''.join(random.choices(value['data_range'], k=value['len']))
            elif key == "bool":
                tmp = random.choice([True, False])
            elif key == "date":
                start_date = datetime.datetime.strptime(value['data_range'][0], '%Y-%m-%d')
                end_date = datetime.datetime.strptime(value['data_range'][1], '%Y-%m-%d')
                time_delta = end_date - start_date
                random_days = random.randint(0, time_delta.days)
                tmp = (start_date + datetime.timedelta(days=random_days)).strftime('%Y-%m-%d')
            elif key == "list":
                item_len = value.get("len", 5)
                tmp = [random.randint(value["data_range"][0], value["data_range"][1]) for _ in range(item_len)]
            elif key == "tuple":
                item_len = value.get("len", 3)
                char_len = value.get("char_len", 2)
                tmp = tuple(''.join(random.choices(value["data_range"], k=char_len)) for _ in range(item_len))
            # 自定义类型 递归处理
            else:
                tmp = data_sampler(1, **value)[0]
            element.append(tmp)
        result.append(element)
    return result

# --- 参数化修饰器 ---
def analyze_statistics(*metrics):
    """参数化修饰器，用于对生成的数据进行统计分析。"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 调用被修饰函数获取数据
            data = func(*args, **kwargs)
            print("--- 生成的随机结构化数据 ---")
            pprint.pprint(data, indent=2)

            # 递归展开所有数值
            def flatten(items):
                flattened = []
                if isinstance(items, dict): items = items.values()
                for item in items:
                    if isinstance(item, (list, tuple, dict)):
                        flattened.extend(flatten(item))
                    elif isinstance(item, (int, float)):
                        flattened.append(item)
                return flattened

            all_values = flatten(data)
            results = {}
            n = len(all_values)
            if n == 0:
                print("\n数据中未找到可供统计的数值。")
                return {metric: None for metric in metrics}

            # 计算统计指标
            sum_val = sum(all_values)
            avg_val = sum_val / n
            variance = sum((x - avg_val) ** 2 for x in all_values) / (n - 1) if n > 1 else 0.0
            rmse = math.sqrt(sum(x ** 2 for x in all_values) / n)

            # 根据参数填充结果
            for metric in metrics:
                if metric.upper() == 'SUM': results['SUM'] = sum_val
                elif metric.upper() == 'AVG': results['AVG'] = avg_val
                elif metric.upper() == 'VAR': results['VAR'] = variance
                elif metric.upper() == 'RMSE': results['RMSE'] = rmse
            return results
        return wrapper
    return decorator

# 应用修饰器
@analyze_statistics('SUM', 'AVG', 'VAR', 'RMSE')
def decorated_data_sampler(*args, **kwargs):
    return data_sampler(*args, **kwargs)

if __name__ == "__main__":
    entry_structure = {
        "int": {"data_range": (1, 10)},
        "float": {"data_range": (0, 100), "decimal_places": 2},
        "student_participant": {
            "str": {"data_range": string.ascii_uppercase, "len": 2},
            "int": {"data_range": (2024001, 2024999)},
            "advisor": {
                "str": {"data_range": string.ascii_uppercase, "len": 3},
                "contact_information": {
                    "int": {"data_range": (1000000, 9999999)},
                    "float": {"data_range": (90.0, 100.0)} # 添加一个float用于统计
                }
            }
        }
    }

    stats = decorated_data_sampler(num=5, **entry_structure)

    print("\n--- 最终统计结果 ---")
    pprint.pprint(stats)