import time
import random
import copy

random.seed(42)

tuple_matrix = tuple(tuple(random.randint(1, 100) for _ in range(10000)) for _ in range(10000))
list_matrix = copy.deepcopy([list(row) for row in tuple_matrix])

# 测试 tuple 矩阵修改时间（由于 tuple 不可变，需要重新构造整个行）
start_time = time.time()
for _ in range(10000):
    row = random.randint(0, 9999)
    col = random.randint(0, 9999)
    new_value = random.randint(1, 100)
    row_list = list(tuple_matrix[row])
    row_list[col] = new_value
    row_tuple = tuple(row_list)
    tuple_matrix = tuple(row_tuple if i == row else tuple_matrix[i] for i in range(len(tuple_matrix)))
tuple_time = time.time() - start_time

# 测试 list 矩阵修改时间
start_time = time.time()
for _ in range(10000):
    row = random.randint(0, 9999)
    col = random.randint(0, 9999)
    new_value = random.randint(1, 100)
    list_matrix[row][col] = new_value
list_time = time.time() - start_time

print(f"Tuple 矩阵修改 10000 次时间消耗：{tuple_time} 秒")
print(f"List 矩阵修改 10000 次时间消耗：{list_time} 秒")