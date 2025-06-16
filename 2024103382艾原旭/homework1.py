import time
import numpy as np
from typing import Union


def create_matrix(container_type: Union[tuple, list], rows: int, cols: int) -> Union[tuple, list]:

    if container_type is tuple:
        return tuple(tuple(0 for _ in range(cols)) for _ in range(rows))
    elif container_type is list:
        return [[0 for _ in range(cols)] for _ in range(rows)]
    else:
        raise ValueError("容器类型必须是tuple或list")


def modify_matrix(matrix: Union[tuple, list], rounds: int) -> float:
    start_time = time.time()

    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    for _ in range(rounds):
        i, j = np.random.randint(0, rows), np.random.randint(0, cols)

        if isinstance(matrix, list):
            matrix[i][j] = 1
        elif isinstance(matrix, tuple):
            temp_list = [list(row) for row in matrix]
            temp_list[i][j] = 1
            matrix = tuple(tuple(row) for row in temp_list)

    return time.time() - start_time


def performance_test():
    size = 100
    rounds = 10000

    print(f"创建{size}x{size}矩阵并执行{rounds}次修改测试...")

    list_matrix = create_matrix(list, size, size)
    list_time = modify_matrix(list_matrix, rounds)
    print(f"列表修改耗时: {list_time:.4f}秒")

    tuple_matrix = create_matrix(tuple, size, size)
    tuple_time = modify_matrix(tuple_matrix, rounds)
    print(f"元组修改耗时: {tuple_time:.4f}秒")

    print(f"\n性能比较: 列表比元组快 {tuple_time / list_time:.1f} 倍")


if __name__ == "__main__":
    performance_test()