import time
import sys

# 设置较小的矩阵大小以避免内存溢出
SIZE = 10000  # 使用10000x10000的矩阵
ITERATIONS = 100000  # 修改次数

print("开始创建数据结构...")

# 创建list矩阵
matrix_list = [[0 for _ in range(SIZE)] for _ in range(SIZE)]

# 创建tuple矩阵
matrix_tuple = tuple(tuple(0 for _ in range(SIZE)) for _ in range(SIZE))

print("数据结构创建完成，开始测试...")

# 测试list修改性能
print("测试list修改性能...")
start_time_l = time.time()
for i in range(ITERATIONS):
    matrix_list[0][0] = i  # 只修改第一个元素
end_time_l = time.time()
list_time = end_time_l - start_time_l

# 测试tuple修改性能
print("测试tuple修改性能...")
start_time_t = time.time()
for i in range(ITERATIONS):
    # 通过创建新的tuple来"修改"第一个元素
    first_row = list(matrix_tuple[0])
    first_row[0] = i
    matrix_tuple = (tuple(first_row),) + matrix_tuple[1:]
end_time_t = time.time()
tuple_time = end_time_t - start_time_t

# 输出结果
print("\n性能测试结果:")
print(f"List修改 {ITERATIONS} 次耗时: {list_time:.6f} 秒")
print(f"Tuple修改 {ITERATIONS} 次耗时: {tuple_time:.6f} 秒")
print(f"Tuple比List慢: {tuple_time/list_time:.2f} 倍")

# 输出内存使用情况
list_size = sys.getsizeof(matrix_list)
tuple_size = sys.getsizeof(matrix_tuple)
print(f"\n内存占用情况:")
print(f"List内存占用: {list_size/1024/1024:.2f} MB")
print(f"Tuple内存占用: {tuple_size/1024/1024:.2f} MB")
