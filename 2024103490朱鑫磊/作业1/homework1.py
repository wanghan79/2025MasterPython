import time
import random

N = 1000
NUM_CHANGES = 10000

# 构造 list 矩阵
list_matrix = [[0 for _ in range(N)] for _ in range(N)]

start = time.time()
for _ in range(NUM_CHANGES):
    i, j = random.randint(0, N - 1), random.randint(0, N - 1)
    list_matrix[i][j] = 1
end = time.time()
print(f"List 修改耗时: {end - start:.2f} 秒")

# 构造 tuple 矩阵
tuple_matrix = tuple(tuple(0 for _ in range(N)) for _ in range(N))

start = time.time()
for _ in range(NUM_CHANGES):
    i, j = random.randint(0, N - 1), random.randint(0, N - 1)
    row = list(tuple_matrix[i]) 
    row[j] = 1
    tuple_matrix = tuple_matrix[:i] + (tuple(row),) + tuple_matrix[i+1:]
end = time.time()
print(f"Tuple 修改耗时: {end - start:.2f} 秒")
