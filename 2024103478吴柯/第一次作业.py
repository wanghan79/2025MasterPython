import time
import random

list_matrix = [[0 for _ in range(10000)] for _ in range(10000)]
tuple_matrix = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))

# 修改 list 矩阵
start_list = time.time()
for _ in range(10000):
    i = random.randint(0, 9999)
    j = random.randint(0, 9999)
    list_matrix[i][j] = random.randint(0, 100)
end_list = time.time()

# 修改 tuple 矩阵
start_tuple = time.time()
for _ in range(10000):
    i = random.randint(0, 9999)
    j = random.randint(0, 9999)
    value = random.randint(0, 100)
    new_row = tuple_matrix[i][:j] + (value,) + tuple_matrix[i][j+1:]
    tuple_matrix = tuple_matrix[:i] + (new_row,) + tuple_matrix[i+1:]
end_tuple = time.time()

print(f"List matrix modify time: {end_list - start_list:.4f} seconds")
print(f"Tuple matrix modify time: {end_tuple - start_tuple:.4f} seconds")
# List matrix modify time: 0.0228 seconds
# Tuple matrix modify time: 6.5508 seconds
