import time
import random

# 初始化10000×10000矩阵
n = 10000
matrix_list = [[0] * n for _ in range(n)]
matrix_tuple = tuple(tuple(row) for row in matrix_list)

# 生成10000个随机修改位置
positions = [(random.randint(0, n-1), random.randint(0, n-1))
             for _ in range(10000)]

# 测试列表
start_time = time.time()
for i, j in positions:
    matrix_list[i][j] = 1
list_time = time.time() - start_time
print(f"list cost: {list_time:.4f} sec")

# 测试元组
start_time = time.time()
current = matrix_tuple
for idx, (i, j) in enumerate(positions):
    row_list = list(current[i])
    row_list[j] = 1
    new_row = tuple(row_list)
    current = current[:i] + (new_row,) + current[i+1:]
tuple_time = time.time() - start_time
print(f"tuple cost: {tuple_time:.4f} sec")

# 结果对比
print(f"\ntuple/list: {tuple_time/list_time:.1f}")

# result:
# list cost: 0.0030 sec
# tuple cost: 2.2739 sec
# tuple/list: 764.5
