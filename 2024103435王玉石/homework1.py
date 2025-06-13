import time
import numpy as np

# 创建一个 10000×10000 的矩阵
rows, cols = 10000, 10000

# 使用 list 创建矩阵
matrix_list = [[0] * cols for _ in range(rows)]

# 使用 tuple 创建矩阵
matrix_tuple = tuple(tuple(0 for _ in range(cols)) for _ in range(rows))

# 测试 list 修改时间
start_time = time.time()
for i in range(10000):
    matrix_list[i % rows][i % cols] = 1  # 修改一个元素
list_time = time.time() - start_time

# 测试 tuple 修改时间
start_time = time.time()
# 对于 tuple 每次修改必须重新创建一个新的 tuple，模拟修改
matrix_tuple = list(matrix_tuple)  # 转换为 list 以便进行修改
for i in range(10000):
    matrix_tuple[i % rows] = tuple(matrix_tuple[i % rows])  # 需要替换行
    row = list(matrix_tuple[i % rows])
    row[i % cols] = 1
    matrix_tuple[i % rows] = tuple(row)  # 修改一个元素并重新生成 tuple
tuple_time = time.time() - start_time

# 输出结果
print(f"List修改时间: {list_time:.6f}秒")
print(f"Tuple修改时间: {tuple_time:.6f}秒")
