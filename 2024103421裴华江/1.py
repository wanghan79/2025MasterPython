
import time
import sys

# 设置较小的矩阵大小以避免内存溢出
MATRIX_SIZE = 10000  # 使用10000x10000的矩阵
MODIFY_TIMES = 100000  # 修改次数

print("开始创建数据结构...")

# 创建list矩阵
list_matrix = [[0 for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]

# 创建tuple矩阵
tuple_matrix = tuple(tuple(0 for _ in range(MATRIX_SIZE)) for _ in range(MATRIX_SIZE))

print("数据结构创建完成，开始测试...")

# 测试list修改性能
print("测试list修改性能...")
start_time_list = time.time()
for idx in range(MODIFY_TIMES):
    list_matrix[0][0] = idx  # 只修改第一个元素
end_time_list = time.time()
list_modify_time = end_time_list - start_time_list

# 测试tuple修改性能
print("测试tuple修改性能...")
start_time_tuple = time.time()
for idx in range(MODIFY_TIMES):
    # 通过创建新的tuple来"修改"第一个元素
    first_row_list = list(tuple_matrix[0])
    first_row_list[0] = idx
    tuple_matrix = (tuple(first_row_list),) + tuple_matrix[1:]
end_time_tuple = time.time()
tuple_modify_time = end_time_tuple - start_time_tuple

# 输出结果
print("\n性能测试结果:")
print(f"List修改 {MODIFY_TIMES} 次耗时: {list_modify_time:.6f} 秒")
print(f"Tuple修改 {MODIFY_TIMES} 次耗时: {tuple_modify_time:.6f} 秒")
print(f"Tuple比List慢: {tuple_modify_time/list_modify_time:.2f} 倍")

# 输出内存使用情况
list_memory = sys.getsizeof(list_matrix)
tuple_memory = sys.getsizeof(tuple_matrix)
print(f"\n内存占用情况:")
print(f"List内存占用: {list_memory/1024/1024:.2f} MB")
print(f"Tuple内存占用: {tuple_memory/1024/1024:.2f} MB")
