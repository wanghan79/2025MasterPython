import time
import random
import copy


N = 10000


list_matrix = [[0 for _ in range(N)] for _ in range(N)]


tuple_matrix = tuple(tuple(0 for _ in range(N)) for _ in range(N))


list_matrix_copy = copy.deepcopy(list_matrix)
start_time = time.time()
for _ in range(10000):
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    list_matrix_copy[i][j] = random.randint(1, 100)
list_time = time.time() - start_time

print(f"List 10000次修改耗时：{list_time:.4f} 秒")


tuple_matrix_copy = copy.deepcopy(tuple_matrix)
start_time = time.time()
for _ in range(10000):
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)

    row_list = list(tuple_matrix_copy[i])
    row_list[j] = random.randint(1, 100)
    new_row = tuple(row_list)

    tmp_list = list(tuple_matrix_copy)
    tmp_list[i] = new_row
    tuple_matrix_copy = tuple(tmp_list)
tuple_time = time.time() - start_time

print(f"Tuple 10000次修改耗时：{tuple_time:.4f} 秒")
