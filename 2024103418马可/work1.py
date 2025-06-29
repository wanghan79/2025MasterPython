import time

# 生成10000x10000的矩阵
def generate_matrix(size):
    return [[0 for _ in range(size)] for _ in range(size)]

# 修改矩阵中的每个元素
def modify_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] += 1

# 生成10000x10000的矩阵并用列表存储
start_time = time.time()
matrix_list = generate_matrix(10000)
end_time = time.time()
print(f"生成列表矩阵时间: {end_time - start_time:.2f} 秒")

# 修改列表矩阵
start_time = time.time()
modify_matrix(matrix_list)
end_time = time.time()
print(f"修改列表矩阵时间: {end_time - start_time:.2f} 秒")

# 生成10000x10000的矩阵并用元组存储
start_time = time.time()
matrix_tuple = tuple(tuple(row) for row in generate_matrix(10000))
end_time = time.time()
print(f"生成元组矩阵时间: {end_time - start_time:.2f} 秒")

# 修改元组矩阵（实际上是生成一个新的元组）
start_time = time.time()
modified_matrix_tuple = tuple(tuple(matrix_tuple[i][j] + 1 for j in range(len(matrix_tuple[i]))) for i in range(len(matrix_tuple)))
end_time = time.time()
print(f"修改元组矩阵时间: {end_time - start_time:.2f} 秒")
