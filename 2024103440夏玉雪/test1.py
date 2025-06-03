import time
import random
import gc

# 参数配置
matrix_size = 10000
modification_rounds = 10000

# 创建可变矩阵（二维 list）
mutable_matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]

# 创建不可变矩阵（tuple of tuples）
readonly_matrix = tuple(tuple(0 for _ in range(matrix_size)) for _ in range(matrix_size))

# 清理垃圾回收
gc.collect()

# 修改 list 矩阵
t1 = time.time()
for _ in range(modification_rounds):
    i, j = random.choices(range(matrix_size), k=2)
    mutable_matrix[i][j] = random.randint(1, 100)
t2 = time.time()
list_duration = t2 - t1

# 修改 tuple 矩阵（每次都要重建对应行和整体结构）
t3 = time.time()
for _ in range(modification_rounds):
    i, j = random.choices(range(matrix_size), k=2)
    updated_row = list(readonly_matrix[i])
    updated_row[j] = random.randint(1, 100)
    readonly_matrix = readonly_matrix[:i] + (tuple(updated_row),) + readonly_matrix[i+1:]
t4 = time.time()
tuple_duration = t4 - t3

# 输出结果
print(f"[list] 修改耗时为：{list_duration:.3f} 秒（matrix_size={matrix_size}, rounds={modification_rounds}）")
print(f"[tuple] 修改耗时为：{tuple_duration:.3f} 秒（matrix_size={matrix_size}, rounds={modification_rounds}）")

