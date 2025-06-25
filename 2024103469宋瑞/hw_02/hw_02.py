"""
数据挖掘-王晗 第二次作业 2025年3月25日
函数功能：
    1、生成随机数据
    2、传入参数包括生成随机数据的数量和字典格式的数据结构（即说明书）
    3、自定义数据结构，不论结构如何，函数都能正确接收和实例化
    4、最终返回一个列表，包含正确数量的数据
    5、旨在理解python动态参数的灵活性
    6、提示：递归处理嵌套数据
"""
import random
import string
import datetime
import pprint

def data_sampler(num, **kwargs):
    result = []
    for _ in range(num):
        element = []
        for key, value in kwargs.items():
            # int类型 给定范围内随机生成
            if key == "int":
                tmp = random.randint(value['data_range'][0], value['data_range'][1])
            # float类型 给定范围内随机生成，指定保留小数点位数
            elif key == "float":
                decimal_places = value.get('decimal_places', 2)
                tmp = round(random.uniform(value['data_range'][0], value['data_range'][1]), decimal_places)
            # string类型 给定范围内随机生成指定数目的字符
            elif key == "str":
                tmp = ''.join(random.choices(value['data_range'], k=value['len']))
            # bool类型 随机生成True或False
            elif key == "bool":
                tmp = random.choice([True, False])
            # date类型 在给定日期范围内随机生成
            elif key == "date":
                start_date = datetime.datetime.strptime(value['data_range'][0], '%Y-%m-%d')
                end_date = datetime.datetime.strptime(value['data_range'][1], '%Y-%m-%d')
                time_delta = end_date - start_date
                random_days = random.randint(0, time_delta.days)
                tmp = (start_date + datetime.timedelta(days=random_days)).strftime('%Y-%m-%d')
            # list类型 根据指定规则生成列表
            elif key == "list":
                item_len = value.get("len", 5)
                tmp = [random.randint(value["data_range"][0], value["data_range"][1]) for _ in range(item_len)]
            # tuple类型 根据指定规则生成元组
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

def main_test():
    """
    自定义数据结构:
    参赛作品
        作品编号(int)
        作品得分(float)
        是否获奖(bool)
        提交日期(date)
        评分列表(list)
        标签(tuple)
        参赛学生
            学生姓名(str)
            学号(int)
            指导教师
                教师姓名(str)
                联系方式
                    电话(int)
                    地址(str)
    """
    entry_structure = {
        "int": {"data_range": (1, 10)},
        "float": {"data_range": (0, 100), "decimal_places": 2},
        "bool": {},
        "date": {"data_range": ["2024-01-01", "2025-12-31"]},
        "list": {"data_range": [80, 100], "len": 5},
        "tuple": {"data_range": string.ascii_lowercase, "len": 3, "char_len": 4},
        "student_participant": {
            "str": {"data_range": string.ascii_uppercase, "len": 2},
            "int": {"data_range": (2024001, 2024999)},
            "advisor": {
                "str": {"data_range": string.ascii_uppercase, "len": 3},
                "contact_information": {
                    "int": {"data_range": (1000000, 9999999)},
                    "str": {"data_range": string.ascii_lowercase, "len": 15}
                }
            }
        }
    }

    def adapt_structure(spec):
        adapted_spec = {}
        for k, v in spec.items():
            if isinstance(v, dict) and "key" in v:
                new_key = v.pop("key")
                adapted_spec[new_key] = adapt_structure(v) if new_key not in ["list", "tuple"] else v
            else:
                adapted_spec[k] = v
        return adapted_spec

    final_structure = adapt_structure(entry_structure)

    result = data_sampler(3, **final_structure)

    print("--- 生成的随机结构化数据 ---")
    pprint.pprint(result, indent=2)

if __name__ == "__main__":
    main_test()