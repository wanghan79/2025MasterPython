# coding=utf-8
import random
import time

n = 10000  # 矩阵维度
trials = 10000  # 修改轮数

# 预生成随机坐标 (避免计入时间)
positions = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(trials)]

# ===== 列表测试 =====
list_matrix = [[0] * n for _ in range(n)]

start_time = time.time()
for i, j in positions:
    list_matrix[i][j] = 1  # 直接修改元素
list_duration = time.time() - start_time

# ===== 元组测试 =====
tuple_matrix = [tuple(0 for _ in range(n)) for _ in range(n)]  # 外层列表+内层元组

start_time = time.time()
for i, j in positions:
    # 重建整行元组实现"修改"
    row_list = list(tuple_matrix[i])
    row_list[j] = 1
    tuple_matrix[i] = tuple(row_list)
tuple_duration = time.time() - start_time

# ===== 结果输出 =====
print(f"列表修改耗时: {list_duration:.4f} 秒")
print(f"元组修改耗时: {tuple_duration:.4f} 秒")
print(f"元组操作比列表慢 {tuple_duration/list_duration:.1f} 倍")