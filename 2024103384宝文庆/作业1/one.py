import time
import random

# 创建一个 10000x10000 的 list
list_matrix = [[0] * 10000 for _ in range(10000)]

# 创建一个 10000x10000 的 tuple
tuple_matrix = tuple([tuple([0] * 10000) for _ in range(10000)])

# 修改 list 的测试
start_time = time.time()
for _ in range(10000):
    i, j = random.randint(0, 9999), random.randint(0, 9999)
    list_matrix[i][j] = 1
list_duration = time.time() - start_time
print(f"List 修改耗时: {list_duration:.2f} 秒")

# 修改 tuple 的测试
start_time = time.time()
for _ in range(10000):
    i, j = random.randint(0, 9999), random.randint(0, 9999)
    row = list(tuple_matrix[i])
    row[j] = 1
    tuple_matrix = tuple(tuple_matrix[:i] + (tuple(row),) + tuple_matrix[i+1:])
tuple_duration = time.time() - start_time
print(f"Tuple 修改耗时: {tuple_duration:.2f} 秒")
