import time
matrix_list = [[0 for i in range(100000)] for j in range(100000)]
matrix_tuple = tuple(tuple(0 for i in range(100000)) for j in range(100000))

start_time_l = time.time()
for i in range(100000):
    matrix_list[0][0]=1
end_time_l = time.time()

start_time_t = time.time()
for i in range(100000):
    matrix_tuple = tuple(tuple(1 if (i == 0 and j == 0) else x for j,x in enumerate(row) )
                         for i,row in enumerate(matrix_tuple))
end_time_t = time.time()

print(f"list时间：{end_time_l-start_time_l:.6f}秒")
print(f"tuple时间：{end_time_t-start_time_t:.6f}秒")