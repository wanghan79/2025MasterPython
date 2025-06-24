import time
import random
import sys

# 调整规模为1000×1000（1百万元素）以在普通计算机上运行
ROWS, COLS = 1000, 1000
NUM_MODIFICATIONS = 1000  # 减少修改次数


def create_list_matrix():
    """创建列表矩阵（可变结构）"""
    print(f"创建 {ROWS}×{COLS} 列表矩阵...")
    return [[0] * COLS for _ in range(ROWS)]


def create_tuple_matrix():
    """创建元组矩阵（不可变结构）"""
    print(f"创建 {ROWS}×{COLS} 元组矩阵...")
    return tuple(tuple(0 for _ in range(COLS)) for _ in range(ROWS))


# 生成随机修改位置
print("生成随机修改位置...")
mod_positions = [
    (random.randint(0, ROWS - 1), random.randint(0, COLS - 1))
    for _ in range(NUM_MODIFICATIONS)
]

# === 列表测试 ===
list_matrix = create_list_matrix()

list_start = time.time()
for i, j in mod_positions:
    # 直接修改元素（O(1)操作）
    list_matrix[i][j] = 1
list_duration = time.time() - list_start

# === 元组测试 ===
tuple_matrix = create_tuple_matrix()

tuple_start = time.time()
for cnt, (i, j) in enumerate(mod_positions):
    # 重建整个矩阵（O(n²)操作）
    new_matrix = []
    for r_idx, row in enumerate(tuple_matrix):
        if r_idx == i:
            # 创建新行并修改元素
            new_row = list(row)
            new_row[j] = 1
            new_matrix.append(tuple(new_row))
        else:
            new_matrix.append(row)
    tuple_matrix = tuple(new_matrix)

    # 显示进度
    if (cnt + 1) % 100 == 0:
        print(f"元组修改进度: {cnt + 1}/{NUM_MODIFICATIONS}")
tuple_duration = time.time() - tuple_start

# 打印结果
print(f"\n测试结果:")
print(f"列表修改时间: {list_duration:.6f} 秒 ({NUM_MODIFICATIONS}次修改)")
print(f"元组修改时间: {tuple_duration:.6f} 秒 ({NUM_MODIFICATIONS}次修改)")
print(f"性能差异: {tuple_duration / list_duration:.1f}倍")

# 内存使用对比
list_memory = sys.getsizeof(list_matrix) + sum(sys.getsizeof(row) for row in list_matrix)
tuple_memory = sys.getsizeof(tuple_matrix) + sum(sys.getsizeof(row) for row in tuple_matrix)
print(f"\n内存占用:")
print(f"列表: {list_memory / (1024 * 1024):.2f} MB")
print(f"元组: {tuple_memory / (1024 * 1024):.2f} MB")