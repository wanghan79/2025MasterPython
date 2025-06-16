import time
import random
import numpy as np


def test_list_performance():
    print("开始测试列表性能...")
    # 创建 10000x10000 的列表矩阵
    matrix = [[0 for _ in range(10000)] for _ in range(10000)]

    start_time = time.time()
    for _ in range(10000):
        i = random.randint(0, 9999)
        j = random.randint(0, 9999)
        matrix[i][j] = 1

    end_time = time.time()
    return end_time - start_time


def test_tuple_performance():
    print("开始测试元组性能...")
    matrix = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))

    start_time = time.time()

    for _ in range(10000):
        i = random.randint(0, 9999)
        j = random.randint(0, 9999)
        temp_list = list(matrix)
        temp_list[i] = list(temp_list[i])
        temp_list[i][j] = 1
        temp_list[i] = tuple(temp_list[i])
        matrix = tuple(temp_list)

    end_time = time.time()
    return end_time - start_time


def main():
    print("Python 可变和不可变数据结构性能测试")
    print("=" * 50)

    list_time = test_list_performance()
    print(f"列表修改耗时: {list_time:.2f} 秒")

    tuple_time = test_tuple_performance()
    print(f"元组修改耗时: {tuple_time:.2f} 秒")

    ratio = tuple_time / list_time
    print(f"\n性能对比：元组修改速度是列表的 {ratio:.2f} 倍")

    print("\n结论：")
    print("1. 列表（list）是可变的，可以直接修改元素，性能较好")
    print("2. 元组（tuple）是不可变的，每次修改都需要重建整个数据结构，性能较差")
    print("3. 在这个测试中，元组的修改速度比列表慢很多倍")


if __name__ == "__main__":
    main()