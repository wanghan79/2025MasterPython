import time
import random

def create_matrix(rows, cols, data_type):
    """创建一个row×cols的矩阵，用指定的数据类型封装"""
    if data_type == tuple:
        # 为tuple创建一个不可变的矩阵表示
        return tuple(tuple(random.randint(0, 100) for _ in range(cols)) for _ in range(rows))
    elif data_type == list:
        # 为list创建可变的矩阵
        return [[random.randint(0, 100) for _ in range(cols)] for _ in range(rows)]
    else:
        raise ValueError("Unsupported data type")

def modify_element(matrix, row, col, new_value, data_type):
    """修改矩阵中的元素"""
    if data_type == list:
        matrix[row][col] = new_value
    elif data_type == tuple:
        # 元组不可变，需要重建整个元组
        matrix_list = list(matrix)
        row_list = list(matrix_list[row])  # 这里修复了索引
        row_list[col] = new_value
        matrix_list[row] = tuple(row_list)
        return tuple(matrix_list)
    else:
        raise ValueError("Unsupported data type")

# 创建10000×10000的list矩阵
print("创建list矩阵...")
start_time = time.time()
list_matrix = create_matrix(10000, 10000, list)
list_creation_time = time.time() - start_time
print(f"list矩阵创建完成，耗时: {list_creation_time:.2f}秒")

# 创建10000×10000的tuple矩阵（注意：这只是创建了元组的表示，实际操作中修改会很慢）
print("创建tuple矩阵...")
start_time = time.time()
tuple_matrix = create_matrix(10000, 10000, tuple)
tuple_creation_time = time.time() - start_time
print(f"tuple矩阵创建完成，耗时: {tuple_creation_time:.2f}秒")

# 修改list矩阵10000次
print("开始修改list矩阵...")
start_time = time.time()
for _ in range(10000):
    row = random.randint(0, 9999)
    col = random.randint(0, 9999)
    modify_element(list_matrix, row, col, random.randint(0, 100), list)
    # 每1000次打印一次进度
    if _ % 1000 == 0 and _ > 0:
        print(f"已完成{int(_/1000)}0%")
list_modification_time = time.time() - start_time
print(f"list矩阵修改完成，耗时: {list_modification_time:.2f}秒")

# 修改tuple矩阵10000次（警告：这个操作会非常慢，可能需要数分钟甚至数小时）
print("开始修改tuple矩阵...")
start_time = time.time()
for _ in range(10000):
    row = random.randint(0, 9999)
    col = random.randint(0, 9999)
    new_value = random.randint(0, 100)
    tuple_matrix = modify_element(tuple_matrix, row, col, new_value, tuple)
    # 每100次打印一次进度
    if _ % 100 == 0 and _ > 0:
        print(f"已完成{int(_/100)}%")
tuple_modification_time = time.time() - start_time
print(f"tuple矩阵修改完成，耗时: {tuple_modification_time:.2f}秒")

# 性能对比总结
print("\n性能对比总结:")
print(f"list创建时间: {list_creation_time:.2f}秒")
print(f"tuple创建时间: {tuple_creation_time:.2f}秒")
print(f"list修改10000次时间: {list_modification_time:.2f}秒")
print(f"tuple修改10000次时间: {tuple_modification_time:.2f}秒")