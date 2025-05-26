import time
import random

# 参数定义
DIMENSION = 1000
NUM_WRITES = 10000

# 构建可变数据结构（二维列表）
mutable_grid = [[0] * DIMENSION for _ in range(DIMENSION)]

# 构建不可变数据结构（嵌套元组）
frozen_grid = tuple(tuple(0 for _ in range(DIMENSION)) for _ in range(DIMENSION))

# 可变结构写入测试
t0 = time.time()
for _ in range(NUM_WRITES):
    row_idx = random.randint(0, DIMENSION - 1)
    col_idx = random.randint(0, DIMENSION - 1)
    mutable_grid[row_idx][col_idx] = random.randint(1, 100)
t1 = time.time()
time_mutable = t1 - t0

t2 = time.time()
for _ in range(NUM_WRITES):
    row_idx = random.randint(0, DIMENSION - 1)
    col_idx = random.randint(0, DIMENSION - 1)

    modified_row = list(frozen_grid[row_idx])
    modified_row[col_idx] = random.randint(1, 100)
    updated_row = tuple(modified_row)

    frozen_grid = frozen_grid[:row_idx] + (updated_row,) + frozen_grid[row_idx + 1:]
t3 = time.time()
time_immutable = t3 - t2

# 输出结果
print(f"Time taken for mutable structure update:   {time_mutable:.4f} seconds")
print(f"Time taken for immutable structure update: {time_immutable:.4f} seconds")
