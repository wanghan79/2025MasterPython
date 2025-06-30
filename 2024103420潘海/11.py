import time
import random
import sys


def create_list_matrix(rows, cols):
    """创建指定大小的列表矩阵"""
    return [[0 for _ in range(cols)] for _ in range(rows)]


def create_tuple_matrix(rows, cols):
    """创建指定大小的元组矩阵"""
    return tuple(tuple(0 for _ in range(cols)) for _ in range(rows))


def modify_list(matrix, rounds):
    """测试列表的修改性能"""
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    start_time = time.time()
    for _ in range(rounds):
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)
        matrix[i][j] = random.randint(1, 100)
    end_time = time.time()

    return end_time - start_time


def modify_tuple(matrix, rounds):
    """测试元组的修改性能"""
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    start_time = time.time()
    current_matrix = matrix
    for _ in range(rounds):
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)
        # 由于元组不可变，每次修改都需要创建一个新的元组
        new_row = list(current_matrix[i])
        new_row[j] = random.randint(1, 100)
        new_matrix = list(current_matrix)
        new_matrix[i] = tuple(new_row)
        current_matrix = tuple(new_matrix)
    end_time = time.time()

    return end_time - start_time


def main():
    rows = 10000
    cols = 10000
    rounds = 10000

    print(f"测试配置: {rows}×{cols} 的矩阵，进行 {rounds} 轮修改")

    # 创建列表矩阵并测试修改性能
    print("正在测试列表的修改性能...")
    list_matrix = create_list_matrix(rows, cols)
    list_time = modify_list(list_matrix, rounds)
    print(f"列表修改耗时: {list_time:.4f} 秒")

    # 创建元组矩阵并测试修改性能
    print("正在测试元组的修改性能...")
    tuple_matrix = create_tuple_matrix(rows, cols)
    tuple_time = modify_tuple(tuple_matrix, rounds)
    print(f"元组修改耗时: {tuple_time:.4f} 秒")

    # 计算性能差异
    speedup = tuple_time / list_time if list_time > 0 else float('inf')
    print(f"列表的修改速度比元组快约 {speedup:.2f} 倍")

    # 打印内存占用情况
    list_size = sys.getsizeof(list_matrix) + sum(sys.getsizeof(row) for row in list_matrix)
    tuple_size = sys.getsizeof(tuple_matrix) + sum(sys.getsizeof(row) for row in tuple_matrix)
    print(f"列表占用内存: {list_size / (1024 * 1024):.2f} MB")
    print(f"元组占用内存: {tuple_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()