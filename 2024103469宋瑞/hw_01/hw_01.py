"""
数据挖掘-王晗 第一次作业 2025年3月25日
列表（可变）与元组（不可变）的性能对比
"""
import time

def list_performance_test():
    # 初始化一个 10000x10000 的矩阵
    my_list = [[0 for _ in range(10000)] for _ in range(10000)]

    # 记录开始时间
    start_time = time.time()

    # 执行 10000 次将第 3 行第 4 列元素赋值为 1 的操作
    for _ in range(10000):
        my_list[2][3] = 1

    # 记录结束时间
    end_time = time.time()

    # 计算执行时间
    execution_time = end_time - start_time
    print(f"列表版本操作执行完成，耗时: {execution_time:.6f} 秒")

def tuple_performance_test():
    # 初始化一个 10000x10000 的元组
    my_tuple = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))

    # 记录开始时间
    start_time = time.time()

    # 执行 10000 次将第 3 行第 4 列元素赋值为 1 的操作
    for _ in range(10000):
        new_row = list(my_tuple[2])
        new_row[3] = 1
        my_tuple = my_tuple[:2] + (tuple(new_row),) + my_tuple[3:]

    # 记录结束时间
    end_time = time.time()

    # 计算执行时间
    execution_time = end_time - start_time
    print(f"元组版本操作执行完成，耗时: {execution_time:.6f} 秒")

if __name__ == "__main__":
    list_performance_test()
    tuple_performance_test()