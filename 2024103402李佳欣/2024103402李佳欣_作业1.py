import time

rows = cols = modify_rounds = 10000
small_rows = small_cols = 10000 

start_time = time.time()
list_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
list_construct_time = time.time() - start_time


start_time = time.time()
for i in range(modify_rounds):
    r, c = i % rows, i % cols
    list_matrix[r][c] += 1
list_modify_time = time.time() - start_time
print(f"List 修改时间: {list_modify_time:.4f} 秒")


start_time = time.time()
tuple_matrix = tuple(tuple(0 for _ in range(small_cols)) for _ in range(small_rows))
tuple_construct_time = time.time() - start_time

start_time = time.time()
for i in range(modify_rounds):
    r, c = i % small_rows, i % small_cols
    row = list(tuple_matrix[r])
    row[c] += 1
    tuple_matrix = tuple_matrix[:r] + (tuple(row),) + tuple_matrix[r+1:]
tuple_modify_time = time.time() - start_time
print(f"Tuple 修改时间: {tuple_modify_time:.4f} 秒")
