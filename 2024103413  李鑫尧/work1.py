import time
import random

MATRIX_SIZE = 10000
MODIFICATIONS = 10000

start_create_list = time.time()
matrix_list = [[0] * MATRIX_SIZE for _ in range(MATRIX_SIZE)]
create_list_time = time.time() - start_create_list
print(f"列表创建完成，耗时: {create_list_time:.2f}秒")

start_mod_list = time.time()
for _ in range(MODIFICATIONS):
    i = random.randint(0, MATRIX_SIZE - 1)
    j = random.randint(0, MATRIX_SIZE - 1)
    matrix_list[i][j] = 1
list_mod_time = time.time() - start_mod_list
print(f"列表修改完成，耗时: {list_mod_time:.2f}秒")

start_create_tuple = time.time()
matrix_tuple = tuple(tuple(0 for _ in range(MATRIX_SIZE)) for _ in range(MATRIX_SIZE))
create_tuple_time = time.time() - start_create_tuple
print(f"元组创建完成，耗时: {create_tuple_time:.2f}秒")

start_mod_tuple = time.time()
current_matrix = matrix_tuple
for _ in range(MODIFICATIONS):
    i = random.randint(0, MATRIX_SIZE - 1)
    j = random.randint(0, MATRIX_SIZE - 1)

    row_list = list(current_matrix[i])
    row_list[j] = 1

    new_row = tuple(row_list)
    current_matrix = current_matrix[:i] + (new_row,) + current_matrix[i + 1:]

tuple_mod_time = time.time() - start_mod_tuple
print(f"元组修改完成，耗时: {tuple_mod_time:.2f}秒")

# 结果对比
print(f"列表创建时间: {create_list_time:.2f}秒")
print(f"元组创建时间: {create_tuple_time:.2f}秒")
print(f"列表修改时间: {list_mod_time:.2f}秒 (共{MODIFICATIONS}次修改)")
print(f"元组修改时间: {tuple_mod_time:.2f}秒 (共{MODIFICATIONS}次修改)")
print(f"元组修改耗时是列表的 {tuple_mod_time / list_mod_time:.1f} 倍")