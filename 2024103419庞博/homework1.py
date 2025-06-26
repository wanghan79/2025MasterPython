import time
import numpy as np

# 设置测试参数
size = 10000  # 矩阵大小
rounds = 10000  # 修改轮数

# 创建列表矩阵 (10000x10000)
list_matrix = [[0 for _ in range(size)] for _ in range(size)]

# 创建元组矩阵 (10000x10000)
tuple_matrix = tuple(tuple(0 for _ in range(size)) for _ in range(size))

# 测试列表修改性能
start_time = time.time()
for _ in range(rounds):
    i = np.random.randint(0, size)
    j = np.random.randint(0, size)
    list_matrix[i][j] = 1  # 直接修改列表元素
list_time = time.time() - start_time

# 测试元组修改性能
start_time = time.time()
for _ in range(rounds):
    i = np.random.randint(0, size)
    j = np.random.randint(0, size)
    # 元组不可变，必须重建整个元组
    new_row = tuple(x if idx != j else 1 for idx, x in enumerate(tuple_matrix[i]))
    tuple_matrix = tuple(new_row if idx != i else row for idx, row in enumerate(tuple_matrix))
tuple_time = time.time() - start_time

# 输出结果
print(f"列表修改 {rounds} 次耗时: {list_time:.4f} 秒")
print(f"元组修改 {rounds} 次耗时: {tuple_time:.4f} 秒")
print(f"元组耗时是列表的 {tuple_time/list_time:.2f} 倍")