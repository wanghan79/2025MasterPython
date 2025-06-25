
import time
import random
import sys

def create_large_list_matrix(rows, cols, value=0):
    return [[value for _ in range(cols)] for _ in range(rows)]

def create_large_tuple_matrix(rows, cols, value=0):
    return tuple(tuple(value for _ in range(cols)) for _ in range(rows))

def test_list_modification(matrix, rounds):
    rows = len(matrix)
    cols = len(matrix[0])
    start_time = time.time()
    for _ in range(rounds):
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)
        matrix[i][j] = 1  # 修改其中一个值
    return time.time() - start_time

def test_tuple_modification(matrix, rounds):
    rows = len(matrix)
    cols = len(matrix[0])
    start_time = time.time()
    for _ in range(rounds):
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)
        # 修改某个元素，需要重建整行，再重建整个元组
        row = list(matrix[i])
        row[j] = 1
        new_row = tuple(row)
        matrix = matrix[:i] + (new_row,) + matrix[i+1:]
    return time.time() - start_time

if __name__ == "__main__":
    ROWS, COLS, ROUNDS = 1000, 1000, 10000  # 建议用 1000×1000，10000×10000 内存需求约 8GB+，小机子容易爆

    print("正在构造 list 矩阵...")
    list_matrix = create_large_list_matrix(ROWS, COLS)

    print("正在构造 tuple 矩阵...")
    tuple_matrix = create_large_tuple_matrix(ROWS, COLS)

    print("开始测试 list 修改性能...")
    list_time = test_list_modification(list_matrix, ROUNDS)
    print(f"List 修改 {ROUNDS} 次耗时: {list_time:.2f} 秒")

    print("开始测试 tuple 修改性能...")
    tuple_time = test_tuple_modification(tuple_matrix, ROUNDS)
    print(f"Tuple 修改 {ROUNDS} 次耗时: {tuple_time:.2f} 秒")

