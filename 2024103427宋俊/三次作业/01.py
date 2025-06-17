import time
import random
import sys

def main():
    size = 10000  # 矩阵维度
    n_modifications = 10000  # 修改次数

    # 检查内存需求（可选）
    list_size = sys.getsizeof([[0] * size for _ in range(size)])
    print(f"预计内存占用: List矩阵 ≈ {list_size / (1024 ** 2):.1f} MB + Tuple矩阵 ≈ {list_size / (1024 ** 2):.1f} MB")

    # 创建List矩阵 (可变)
    print("创建List矩阵...")
    list_matrix = [[0] * size for _ in range(size)]

    # 创建Tuple矩阵 (不可变)
    print("创建Tuple矩阵...")
    tuple_matrix = tuple(tuple(0 for _ in range(size)) for _ in range(size))

    # 生成随机修改坐标
    random.seed(42)
    modifications = [
        (random.randint(0, size - 1), random.randint(0, size - 1))
        for _ in range(n_modifications)
    ]

    # 测试List修改性能
    print(f"开始List修改 ({n_modifications}次)...")
    start_list = time.time()
    for i, j in modifications:
        list_matrix[i][j] = 1  # 直接修改元素
    list_time = time.time() - start_list
    print(f"List修改完成: {list_time:.4f} 秒")

    # 测试Tuple修改性能
    print(f"开始Tuple修改 ({n_modifications}次)...")
    start_tuple = time.time()
    current = tuple_matrix
    for idx, (i, j) in enumerate(modifications):
        # 重建整个矩阵（每次修改都需要创建新元组）
        new_matrix = []
        for row_idx, row in enumerate(current):
            if row_idx == i:
                new_row = list(row)
                new_row[j] = 1
                new_matrix.append(tuple(new_row))
            else:
                new_matrix.append(row)
        current = tuple(new_matrix)

        # 显示进度（可选）
        if (idx + 1) % 1000 == 0:
            print(f"  进度: {idx + 1}/{n_modifications}")
    tuple_time = time.time() - start_tuple
    print(f"Tuple修改完成: {tuple_time:.4f} 秒")

    # 结果对比
    print("\n性能对比结果:")
    print(f"List (可变) 修改时间: {list_time:.6f} 秒")
    print(f"Tuple (不可变) 修改时间: {tuple_time:.6f} 秒")
    print(f"Tuple 比 List 慢 {tuple_time / list_time:.1f} 倍")
if __name__ == "__main__":
    main()
