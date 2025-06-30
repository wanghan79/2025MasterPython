import time
import random

# 创建 10000x10000 的可变列表矩阵
array_matrix = [[0] * 10000 for _ in range(10000)]

# 创建 10000x10000 的不可变元组矩阵
immutable_matrix = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))

# 测试可变列表的修改性能
start = time.time()
for _ in range(10000):
    x = random.randint(0, 9999)
    y = random.randint(0, 9999)
    array_matrix[x][y] = random.randint(1, 100)  # 直接修改元素
end = time.time()
array_time = end - start

# 测试不可变元组的修改性能
start = time.time()
for _ in range(10000):
    x = random.randint(0, 9999)
    y = random.randint(0, 9999)
    # 需要重建整个数据结构
    temp_row = list(immutable_matrix[x])
    temp_row[y] = random.randint(1, 100)
    immutable_matrix = (
        immutable_matrix[:x] 
        + (tuple(temp_row),) 
        + immutable_matrix[x+1:]
    )
end = time.time()
immutable_time = end - start

print(f"可变列表修改时间: {array_time:.4f} 秒")
print(f"不可变元组修改时间: {immutable_time:.4f} 秒")
