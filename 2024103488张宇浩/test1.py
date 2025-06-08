import time
import random
import copy

# 创建一个 10000x10000 的矩阵数据（用较小值测试可行性，建议先用 100x100）
SIZE = 1000  # 建议先用 1000，10000 可能会 OOM
MODIFY_TIMES = 10000

# 构造 list 矩阵
list_matrix = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

# 构造 tuple 矩阵（元组嵌套元组）
tuple_matrix = tuple(tuple(0 for _ in range(SIZE)) for _ in range(SIZE))

# list 修改测试
start_time = time.time()
for _ in range(MODIFY_TIMES):
    i = random.randint(0, SIZE - 1)
    j = random.randint(0, SIZE - 1)
    list_matrix[i][j] = 1
list_time = time.time() - start_time
print(f"List 修改耗时: {list_time:.4f} 秒")

# tuple 修改测试（每次都要重建）
start_time = time.time()
for _ in range(MODIFY_TIMES):
    i = random.randint(0, SIZE - 1)
    j = random.randint(0, SIZE - 1)

    # 复制并修改指定行
    row = list(tuple_matrix[i])
    row[j] = 1
    new_row = tuple(row)

    # 创建新的矩阵
    tuple_matrix = tuple(
        new_row if idx == i else row_
        for idx, row_ in enumerate(tuple_matrix)
    )
tuple_time = time.time() - start_time
print(f"Tuple 修改耗时: {tuple_time:.4f} 秒")
