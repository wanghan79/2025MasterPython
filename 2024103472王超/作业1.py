import time
import random

# 构造 10000 x 10000 的 list 矩阵
list_matrix = [[0] * 10000 for _ in range(10000)]
# 构造 10000 x 10000 的 tuple 矩阵
tuple_matrix = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))

# 修改 list 的耗时
start_list = time.time()
for _ in range(10000):
    i = random.randint(0, 9999)
    j = random.randint(0, 9999)
    list_matrix[i][j] = 1  # 直接修改
end_list = time.time()

# 修改 tuple 的耗时（必须重建整行）
start_tuple = time.time()
for _ in range(10000):
    i = random.randint(0, 9999)
    j = random.randint(0, 9999)
    row = list(tuple_matrix[i])        # 将 tuple 转为 list
    row[j] = 1                          # 修改
    tuple_matrix = tuple(
        row if x == i else tuple_matrix[x] for x in range(10000)
    )  # 替换整行，重新构造整个矩阵
end_tuple = time.time()

print(f"List 修改耗时: {end_list - start_list:.2f} 秒")
print(f"Tuple 修改耗时: {end_tuple - start_tuple:.2f} 秒")
