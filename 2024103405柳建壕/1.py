import time
import sys
import random


def create_tuple_matrix(rows, cols):
    """创建元组矩阵"""
    return tuple(tuple(0 for _ in range(cols)) for _ in range(rows))


def create_list_matrix(rows, cols):
    """创建列表矩阵"""
    return [[0 for _ in range(cols)] for _ in range(rows)]


def modify_tuple_matrix(matrix, num_iterations):
    """测试元组修改性能"""
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    start_time = time.time()
    current_matrix = matrix

    for _ in range(num_iterations):
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)

        # 元组不可变，创建新元组
        new_row = list(current_matrix[i])
        new_row[j] += 1
        new_matrix = tuple(new_row if k == i else current_matrix[k] for k in range(rows))
        current_matrix = new_matrix

    end_time = time.time()
    return end_time - start_time


def modify_list_matrix(matrix, num_iterations):
    """测试列表修改性能"""
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    start_time = time.time()

    for _ in range(num_iterations):
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)
        matrix[i][j] += 1  # 直接修改列表元素

    end_time = time.time()
    return end_time - start_time


def main():
    rows = 10000
    cols = 10000
    iterations = 10000

    list_matrix = create_list_matrix(rows, cols)
    tuple_matrix = create_tuple_matrix(rows, cols)

    # 测试列表性能
    print("测试列表")
    list_time = modify_list_matrix(list_matrix, iterations)

    # 测试元组性能
    print("测试元组")
    tuple_time = modify_tuple_matrix(tuple_matrix, iterations)

    # 输出结果
    print("\n性能测试结果：")
    print(f"列表修改耗时：{list_time:.4f} 秒")
    print(f"元组修改耗时：{tuple_time:.4f} 秒")
    print(f"元组/列表耗时比：{tuple_time / list_time:.2f} 倍")

    print("\n结论：")
    print("列表是可变的，可以直接修改某个元素，速度很快。")
    print("元组是不可变的，每次修改都要重建整个元组，导致性能极差。")


if __name__ == "__main__":
    main()
