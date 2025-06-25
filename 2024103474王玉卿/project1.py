import time
import random
import copy

ROWS, COLS = 10000, 10000
MODIFY_TIMES = 10000


row_data = [0] * COLS


print("创建 list 矩阵...")
list_matrix = [row_data.copy() for _ in range(ROWS)]
print("创建 tuple 矩阵...")
tuple_matrix = tuple(row_data)
print("修改 list 矩阵...")
start_list = time.time()

for _ in range(MODIFY_TIMES):
    i = random.randint(0, ROWS - 1)
    j = random.randint(0, COLS - 1)
    list_matrix[i][j] = 1
end_list = time.time()

print("修改 tuple 矩阵（模拟）...")
start_tuple = time.time()
for _ in range(MODIFY_TIMES):
    idx = random.randint(0, COLS - 1)
    new_tuple = tuple_matrix[:idx] + (1,) + tuple_matrix[idx+1:]
end_tuple = time.time()

print(f"list 修改耗时: {end_list - start_list:.4f} 秒")
print(f"tuple 模拟修改耗时: {end_tuple - start_tuple:.4f} 秒")
