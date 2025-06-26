import time
import random

def create_matrix(rows, cols, data_type):
   
    if data_type == list:
        return [[random.randint(0, 100) for _ in range(cols)] for _ in range(rows)]
    elif data_type == tuple:
        return tuple(tuple(random.randint(0, 100) for _ in range(cols)) for _ in range(rows))
    else:
        raise ValueError("Unsupported data type.")

def modify_element(matrix, row, col, new_value, data_type):

    if data_type == list:
        matrix[row][col] = new_value
        return matrix
    elif data_type == tuple:
        matrix_list = list(matrix)
        row_list = list(matrix_list[row])
        row_list[col] = new_value
        matrix_list[row] = tuple(row_list)
        return tuple(matrix_list)
    else:
        raise ValueError("Unsupported data type.")

# 设置参数
ROWS, COLS = 10000, 10000
MODIFY_TIMES = 10000

# 创建 list 矩阵
print("正在创建 list 矩阵...")
start = time.time()
list_matrix = create_matrix(ROWS, COLS, list)
list_create_time = time.time() - start
print(f"list 矩阵创建完成，耗时：{list_create_time:.2f} 秒\n")

# 创建 tuple 矩阵
print("正在创建 tuple 矩阵...")
start = time.time()
tuple_matrix = create_matrix(ROWS, COLS, tuple)
tuple_create_time = time.time() - start
print(f"tuple 矩阵创建完成，耗时：{tuple_create_time:.2f} 秒\n")

# 修改 list 矩阵
print("正在修改 list 矩阵...")
start = time.time()
for i in range(MODIFY_TIMES):
    r, c = random.randint(0, ROWS - 1), random.randint(0, COLS - 1)
    list_matrix = modify_element(list_matrix, r, c, random.randint(0, 100), list)
list_modify_time = time.time() - start
print(f"list 矩阵修改完成，耗时：{list_modify_time:.2f} 秒\n")

# 修改 tuple 矩阵
print("正在修改 tuple 矩阵（该过程较慢，请耐心等待）...")
start = time.time()
for i in range(MODIFY_TIMES):
    r, c = random.randint(0, ROWS - 1), random.randint(0, COLS - 1)
    tuple_matrix = modify_element(tuple_matrix, r, c, random.randint(0, 100), tuple)
tuple_modify_time = time.time() - start
print(f"tuple 矩阵修改完成，耗时：{tuple_modify_time:.2f} 秒\n")

# 输出性能对比
print("========== 性能对比结果 ==========")
print(f"list 创建时间         ：{list_create_time:.2f} 秒")
print(f"tuple 创建时间        ：{tuple_create_time:.2f} 秒")
print(f"list 修改 10000 次时间：{list_modify_time:.2f} 秒")
print(f"tuple 修改 10000 次时间：{tuple_modify_time:.2f} 秒")
print("===================================")
