import time

# 初始化一个100x100的矩阵
matrix_list = [[0 for i in range(100)] for i in range(100)]
matrix_tuple = tuple(tuple(0 for i in range(100)) for i in range(100))


start_time_list = time.time()
for i in range(100):
    for j in range(100):
        matrix_list[i][j] = 1
list_time=time.time()-start_time_list
print(f"list_time:{list_time}")

start_time_tuple = time.time()
# 进行 10000 次修改（重新创建元组）
for i in range(10000):
    # 生成一个新的元组，模拟修改操作
    data = tuple(tuple(1 for i in range(100)) for i in range(100))
    matrix_tuple=data
tuple_time=time.time()-start_time_tuple
print("tuple_time:",tuple_time)
