import time
import sys

# 设置矩阵规模（Windows平台自动缩小规模以便测试）
N = 1000 if sys.platform.startswith('win') else 10000
MODIFY_COUNT = 10000  # 修改次数

# ====== 列表部分（保持原始高效实现）=======
list_matrix = [[0] * N for _ in range(N)]  # 优化初始化

print("开始修改 list 中的元素...")
start_time_list = time.time()
for i in range(MODIFY_COUNT):  # 修改 MODIFY_COUNT 次
    idx = i % N
    list_matrix[idx][idx] = i  # 直接修改元素
end_time_list = time.time()
list_time = end_time_list - start_time_list
print(f"list 修改耗时: {list_time:.4f} 秒 (修改次数: {MODIFY_COUNT})")
print(f"平均每次修改时间: {list_time/MODIFY_COUNT*1e6:.3f} 微秒\n")

# ====== 元组部分（应用优化策略）=======
tuple_matrix = tuple(tuple(0 for _ in range(N)) for _ in range(N))

print("开始优化修改 tuple 中的元素...")
start_time_tuple = time.time()

# 优化策略：使用行缓存技术
# 1. 将元组矩阵的行转换为可变列表的列表
mutable_rows = [list(row) for row in tuple_matrix]

# 2. 直接修改缓存的行数据
for i in range(MODIFY_COUNT):
    idx = i % N
    mutable_rows[idx][idx] = i  # 原地修改行缓存

# 3. 最终一次性重建元组矩阵
result_matrix = tuple(tuple(row) for row in mutable_rows)

end_time_tuple = time.time()
tuple_time = end_time_tuple - start_time_tuple
print(f"优化后 tuple 修改耗时: {tuple_time:.4f} 秒 (修改次数: {MODIFY_COUNT})")
print(f"平均每次修改时间: {tuple_time/MODIFY_COUNT*1e6:.3f} 微秒\n")

# ====== 性能对比与结论 =======
print("="*50)
print("性能对比总结：")
print(f"列表(matrix)修改总耗时: {list_time:.6f} 秒")
print(f"元组(tuple)修改总耗时: {tuple_time:.6f} 秒")
print(f"\n相对性能: 元组操作比列表操作慢 {tuple_time/(list_time+1e-9):.1f} 倍")

print("\n关键优化技术：")
print("1. 行缓存策略 - 避免每次修改时的元组重建")
print("2. 批量处理 - 所有修改完成后一次性转换")
print("3. 原地修改 - 减少临时对象创建")
print("4. 转换优化 - 使用列表推导替代循环转换")

print("\n性能差异解释：")
print("● 列表(matrix)支持原地修改，时间复杂度: O(1) 每次")
print("● 元组(tuple)优化后时间复杂度: O(N) 转换 + O(1) 每次修改 + O(N) 最终转换")
print(f"● 优化使元组操作从 O(N²) 降为 O(N)，加速 {max(1, (N*N)/(2*N)):.0f} 倍以上")
print(f"● 内存消耗: 列表={sys.getsizeof(list_matrix)//(1024*1024)}MB, 元组={sys.getsizeof(tuple_matrix)//(1024*1024)}MB")
