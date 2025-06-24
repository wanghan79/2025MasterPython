import time
import random

# 设置矩阵大小
MATRIX_SIZE = 10000
MODIFY_COUNT = 10000

list_matrix = [[0 for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]

tuple_matrix = tuple(tuple(0 for _ in range(MATRIX_SIZE)) for _ in range(MATRIX_SIZE))

start_time_list = time.time()
for _ in range(MODIFY_COUNT):
    i = random.randint(0, MATRIX_SIZE - 1)
    j = random.randint(0, MATRIX_SIZE - 1)
    list_matrix[i][j] = 1
end_time_list = time.time()
list_duration = end_time_list - start_time_list
print(f"修改 list 耗时: {list_duration:.4f} 秒")

start_time_tuple = time.time()
temp_matrix = tuple_matrix
for _ in range(MODIFY_COUNT):
    i = random.randint(0, MATRIX_SIZE - 1)
    j = random.randint(0, MATRIX_SIZE - 1)
    row = list(temp_matrix[i])
    row[j] = 1
    new_row = tuple(row)
    temp_matrix = temp_matrix[:i] + (new_row,) + temp_matrix[i+1:]
end_time_tuple = time.time()
tuple_duration = end_time_tuple - start_time_tuple
print(f"修改 tuple 耗时: {tuple_duration:.4f} 秒")

