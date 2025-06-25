import time
import random

# 参数设置
ROWS = 10000
COLS = 10000
ITERATIONS = 10000

# 创建列表矩阵（可变容器）
list_matrix = [[0 for _ in range(COLS)] for _ in range(ROWS)]

# 创建元组矩阵（不可变容器，嵌套元组结构）
tuple_matrix = tuple(tuple(0 for _ in range(COLS)) for _ in range(ROWS))

# 测试列表修改时间
list_start = time.perf_counter()
for _ in range(ITERATIONS):
    i = random.randint(0, ROWS-1)
    j = random.randint(0, COLS-1)
    list_matrix[i][j] = 1  # 单次修改操作（O(1)时间复杂度）
list_elapsed = time.perf_counter() - list_start

# 测试元组修改时间（需要重建整个元组结构）
tuple_start = time.perf_counter()
current_tuple = tuple_matrix  # 保存当前元组状态
for _ in range(ITERATIONS):
    i = random.randint(0, ROWS-1)
    j = random.randint(0, COLS-1)

    # 逐层重建元组结构（O(ROWS+COLS)时间复杂度）
    row = current_tuple[i]
    new_row = row[:j] + (1,) + row[j+1:]  # 重建单行元组
    current_tuple = current_tuple[:i] + (new_row,) + current_tuple[i+1:]  # 重建整个矩阵元组
tuple_elapsed = time.perf_counter() - tuple_start

# 输出结果对比
print(f"列表修改耗时: {list_elapsed:.4f} 秒")
print(f"元组修改耗时: {tuple_elapsed:.4f} 秒")
print(f"元组修改时间是列表的 {tuple_elapsed / list_elapsed:.2f} 倍")
