import time
import random

# 设置矩阵大小和修改次数
N = 10000
MODIFY_TIMES = 10000

# 生成初始数据
print("正在生成初始数据...")
list_matrix = [[0 for _ in range(N)] for _ in range(N)]
tuple_matrix = tuple(tuple(0 for _ in range(N)) for _ in range(N))

# 随机生成要修改的位置
positions = [(random.randint(0, N-1), random.randint(0, N-1)) for _ in range(MODIFY_TIMES)]

# 测试 list 修改性能
print("开始测试 list 修改性能...")
start_time = time.time()
for i, j in positions:
    list_matrix[i][j] = 1
list_time = time.time() - start_time
print(f"list 修改 {MODIFY_TIMES} 次耗时: {list_time:.4f} 秒")

# 测试 tuple 修改性能
# 由于 tuple 不可变，每次都要新建一行和新建整个矩阵
print("开始测试 tuple 修改性能...")
tuple_matrix_copy = tuple_matrix
start_time = time.time()
for i, j in positions:
    row = list(tuple_matrix_copy[i])
    row[j] = 1
    tuple_matrix_copy = tuple(
        row if idx == i else tuple_matrix_copy[idx]
        for idx in range(N)
    )
tuple_time = time.time() - start_time
print(f"tuple 修改 {MODIFY_TIMES} 次耗时: {tuple_time:.4f} 秒")

