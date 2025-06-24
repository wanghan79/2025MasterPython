import time
import random

N = 1000  # 矩阵大小
MODIFY_TIMES = 10000  # 修改次数

# -------- list --------
print("list...")
list_matrix = [[0] * N for _ in range(N)]

start_time = time.time()
for _ in range(MODIFY_TIMES):
    i, j = random.randint(0, N - 1), random.randint(0, N - 1)
    list_matrix[i][j] = 1
end_time = time.time()
print("list 修改耗时：", end_time - start_time, "秒\n")


# -------- tuple  --------
print("tuple...")
tuple_matrix = tuple([tuple([0] * N) for _ in range(N)])

start_time = time.time()
for _ in range(MODIFY_TIMES):
    i, j = random.randint(0, N - 1), random.randint(0, N - 1)
    row = list(tuple_matrix[i])
    row[j] = 1
    new_row = tuple(row)
    # 重建整个矩阵
    tuple_matrix = tuple(
        new_row if idx == i else tuple_matrix[idx]
        for idx in range(N)
    )
end_time = time.time()
print("tuple 修改耗时：", end_time - start_time, "秒")
