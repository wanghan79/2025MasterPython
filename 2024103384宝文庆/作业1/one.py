import time

matrix_size = 10000

list_matrix = [[0] * matrix_size for _ in range(matrix_size)]
start_time = time.time()
for i in range(matrix_size):
    list_matrix[i][i] = 1
end_time = time.time()
list_time = end_time - start_time


tuple_matrix = tuple(tuple([0] * matrix_size) for _ in range(matrix_size))
start_time = time.time()

tuple_matrix = tuple(tuple([1 if i == j else 0 for j in range(matrix_size)]) for i in range(matrix_size))
end_time = time.time()
tuple_time = end_time - start_time

print(f"List 修改耗时: {list_time:.4f}秒")
print(f"Tuple 修改耗时: {tuple_time:.4f}秒")
