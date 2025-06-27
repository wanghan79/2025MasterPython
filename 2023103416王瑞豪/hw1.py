import time
import random

N = 10000
MODIFY_TIMES = 10000

list_matrix = [[0 for _ in range(N)] for _ in range(N)]

tuple_matrix = tuple(tuple(0 for _ in range(N)) for _ in range(N))

# 修改 list_matrix 的时间测试
start = time.time()
for _ in range(MODIFY_TIMES):
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    list_matrix[i][j] = 1
end = time.time()
print("List 修改耗时：", end - start, "秒")

# 修改 tuple_matrix 的时间测试
start = time.time()
for _ in range(MODIFY_TIMES):
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    row = list(tuple_matrix[i])
    row[j] = 1
    new_row = tuple(row)
    tuple_matrix = tuple(new_row if k == i else tuple_matrix[k] for k in range(N))
end = time.time()
print("Tuple 修改耗时：", end - start, "秒")
