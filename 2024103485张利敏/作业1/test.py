import random
import time

# 10000*10000的tuple
data = tuple(tuple(random.randint(1, 100) for _ in range(10000)) for _ in range(10000))

start_time_tuple = time.time()

for _ in range(10000):
    i = random.randint(0, 9999)
    j = random.randint(0, 9999)
    new_row = list(data[i])
    new_row[j] = random.randint(1, 100)
    new_data = list(data)
    new_data[i] = tuple(new_row)
    data = tuple(new_data)

end_time_tuple = time.time()
sum_time_tuple = end_time_tuple - start_time_tuple

print(f"总共耗时: {sum_time_tuple:.4f} 秒")

# 10000x10000的矩阵，初始值为0
matrix = [[0 for _ in range(10000)] for _ in range(10000)]

start_time_list = time.time()

for _ in range(10000):
    # 随机选择行和列
    row = random.randint(0, 9999)
    col = random.randint(0, 9999)
    # 修改矩阵中的值
    matrix[row][col] = random.randint(1, 100)

end_time_list = time.time()
sum_time_list = end_time_list - start_time_list

# 输出总耗时
print(f"总耗时: {sum_time_list:.4f} 秒")
print(f"List比Tuple快: {sum_time_tuple/sum_time_list:.2f}倍")
