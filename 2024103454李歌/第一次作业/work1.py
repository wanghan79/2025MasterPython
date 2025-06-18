import time
import random

# 设置矩阵尺寸和修改次数
N = 10000
MODIFY_TIMES = 10000

# 初始化 list 矩阵
list_matrix = [[0] * N for _ in range(N)]

# 初始化 tuple 矩阵（注意：元组本身不能修改，需嵌套不可变结构）
tuple_matrix = tuple(tuple(0 for _ in range(N)) for _ in range(N))

# 测试 list 修改时间
start = time.time()
for _ in range(MODIFY_TIMES):
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    list_matrix[i][j] = 1  # 直接修改
end = time.time()
print(f"List 修改耗时: {end - start:.4f} 秒")

# 测试 tuple 修改时间（每次都要复制整行）
start = time.time()
for _ in range(MODIFY_TIMES):
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    row = list(tuple_matrix[i])
    row[j] = 1
    tuple_matrix = tuple(
        row if idx == i else tuple_matrix[idx]
        for idx in range(N)
    )
end = time.time()
print(f"Tuple 修改耗时: {end - start:.4f} 秒")
