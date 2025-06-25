import time
import random
import copy

# 设置矩阵大小和修改次数
N = 10000
MODIFY_TIMES = 10000

# ========== 可变容器 list ==========
print("构造 list 矩阵...")
list_matrix = [[0 for _ in range(N)] for _ in range(N)]

print("开始修改 list 中的元素...")
start_time = time.time()
for _ in range(MODIFY_TIMES):
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    list_matrix[i][j] = 1
end_time = time.time()
print(f"list 修改耗时：{end_time - start_time:.2f} 秒")

# ========== 不可变容器 tuple ==========
print("\n构造 tuple 矩阵...")
tuple_matrix = tuple(tuple(0 for _ in range(N)) for _ in range(N))

print("开始修改 tuple 中的元素（通过重建）...")
start_time = time.time()
for _ in range(MODIFY_TIMES):
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    
    # 修改方式：必须复制原有 tuple，再替换指定元素
    row = list(tuple_matrix[i])         # tuple → list
    row[j] = 1                           # 修改
    new_row = tuple(row)                # list → tuple
    tuple_matrix = tuple_matrix[:i] + (new_row,) + tuple_matrix[i+1:]  # 替换整行
end_time = time.time()
print(f"tuple 修改耗时：{end_time - start_time:.2f} 秒")
