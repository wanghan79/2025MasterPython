import time
import numpy as np

# 创建10000×10000矩阵
list_matrix = np.zeros((10000, 10000)).tolist()  # 可变列表
tuple_matrix = tuple(tuple(row) for row in list_matrix)  # 不可变元组

# 生成10000个随机修改位置
n = 10000
rows = np.random.randint(0, 10000, size=n)
cols = np.random.randint(0, 10000, size=n)

# 测试列表修改时间
start = time.time()
for i, j in zip(rows, cols):
    list_matrix[i][j] = 1  # 直接修改元素
list_time = time.time() - start

# 测试元组修改时间
start = time.time()
for i, j in zip(rows, cols):
    # 必须重建整个元组
    new_row = list(tuple_matrix[i])
    new_row[j] = 1
    tuple_matrix = tuple_matrix[:i] + (tuple(new_row),) + tuple_matrix[i+1:]
tuple_time = time.time() - start

print(f"列表修改时间: {list_time:.4f}秒")
print(f"元组修改时间: {tuple_time:.4f}秒")
print(f"元组修改时间是列表修改时间的 {tuple_time/list_time:.1f}倍")