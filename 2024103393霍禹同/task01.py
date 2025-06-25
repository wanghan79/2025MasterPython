import time

# 初始化10000x10000的list
size = 10000
matrix_list = [[0] * size for _ in range(size)]

# 记录list修改开始时间
start_time_list = time.time()

# 修改所有元素（list）
for i in range(size):
    for j in range(size):
        matrix_list[i][j] = i + j  # 修改所有元素

# 记录list修改结束时间
end_time_list = time.time()

# 计算并输出list修改运行时间
print(f"修改 10000×10000 列表中的所有元素耗时: {end_time_list - start_time_list:.6f} 秒")

# 初始化10000x10000的tuple
matrix_tuple = tuple(tuple(0 for _ in range(size)) for _ in range(size))

# 记录tuple修改开始时间
start_time_tuple = time.time()

# 由于tuple不可变，我们基于原tuple创建新的tuple
new_matrix_tuple = tuple(
    tuple(matrix_tuple[i][j] + (i + j) for j in range(size)) for i in range(size)
)

matrix_tuple = new_matrix_tuple

# 记录tuple修改结束时间
end_time_tuple = time.time()

# 计算并输出tuple修改运行时间
print(f"修改 10000×10000 元组中的所有元素耗时: {end_time_tuple - start_time_tuple:.6f} 秒")
