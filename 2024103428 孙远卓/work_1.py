import time
import random

# 初始化配置
size = 10000
modifications = 10000

# 创建列表矩阵 (可变)
list_matrix = [[0] * size for _ in range(size)]

# 预生成随机修改位置
positions = [(random.randint(0, size-1), random.randint(0, size-1)) for _ in range(modifications)]

# 测试列表修改时间
start = time.time()
for i, j in positions:
    list_matrix[i][j] = 1  # 直接修改元素
list_time = time.time() - start

# 创建元组矩阵 (不可变)
tuple_matrix = tuple(tuple(row) for row in list_matrix)

# 测试元组修改时间
start = time.time()
for i, j in positions:
    # 重建整个矩阵：复制未修改行 + 构建新行
    new_row = tuple(1 if col_idx == j else val for col_idx, val in enumerate(tuple_matrix[i]))
    tuple_matrix = tuple(
        new_row if row_idx == i else row
        for row_idx, row in enumerate(tuple_matrix))
tuple_time = time.time() - start

print(f"List modification time: {list_time:.4f} seconds")
print(f"Tuple modification time: {tuple_time:.4f} seconds")