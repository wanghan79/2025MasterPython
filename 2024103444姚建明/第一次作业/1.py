import time
import random

# 构造 10000×10000 的 list 矩阵
list_matrix = [[0 for _ in range(10000)] for _ in range(10000)]
# 构造 10000×10000 的 tuple 矩阵
tuple_matrix = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))

# list 修改测试
start = time.time()
for _ in range(10000):
    i = random.randint(0, 9999)
    j = random.randint(0, 9999)
    list_matrix[i][j] = 1
end = time.time()
print(f"List 修改 10000 次耗时: {end - start:.2f} 秒")

# tuple 修改测试
start = time.time()
for _ in range(10000):
    i = random.randint(0, 9999)
    j = random.randint(0, 9999)
    # 只能整体替换一行
    row = list(tuple_matrix[i])
    row[j] = 1
    tuple_matrix = tuple(
        row if idx == i else tuple_matrix[idx]
        for idx in range(10000)
    )
end = time.time()
print(f"Tuple 修改 10000 次耗时: {end - start:.2f} 秒")
