import time
import sys


def main():
    size = 10000  # 矩阵维度
    n_modifications = 10000  # 修改次数

    # 检查内存需求（10000x10000矩阵约800MB）
    print(f"创建 {size}x{size} 矩阵 (约需800MB内存)...")
    sys.stdout.flush()

    # 创建列表矩阵（可变）
    start_list = time.time()
    list_matrix = [[0] * size for _ in range(size)]
    list_creation_time = time.time() - start_list
    print(f"列表矩阵创建时间: {list_creation_time:.4f}秒")

    # 创建元组矩阵（不可变）
    start_tuple = time.time()
    tuple_matrix = tuple(tuple(0 for _ in range(size)) for _ in range(size))
    tuple_creation_time = time.time() - start_tuple
    print(f"元组矩阵创建时间: {tuple_creation_time:.4f}秒")

    # 测试列表修改性能
    print(f"\n执行{n_modifications}次列表修改...")
    start = time.time()
    for _ in range(n_modifications):
        # 随机修改一个元素（实际应用中应使用随机坐标）
        list_matrix[0][0] = 1  # 固定位置修改避免随机开销
    list_mod_time = time.time() - start
    print(f"列表修改总时间: {list_mod_time:.4f}秒")
    print(f"平均每次修改: {list_mod_time/n_modifications:.8f}秒")

    # 测试元组修改性能
    print(f"\n执行{n_modifications}次元组修改...")
    start = time.time()
    for _ in range(n_modifications):
        # 重建整个矩阵来"修改"一个元素
        new_matrix = []
        for i, row in enumerate(tuple_matrix):
            if i == 0:  # 修改第一行
                new_row = (1,) + row[1:]  # 替换第一个元素
                new_matrix.append(new_row)
            else:
                new_matrix.append(row)
        tuple_matrix = tuple(new_matrix)  # 重建元组
    tuple_mod_time = time.time() - start
    print(f"元组修改总时间: {tuple_mod_time:.4f}秒")
    print(f"平均每次修改: {tuple_mod_time/n_modifications:.8f}秒")

    # 性能对比
    slowdown = tuple_mod_time / list_mod_time
    print(f"\n性能对比: 元组操作比列表慢约{slowdown:.1f}倍")


if __name__ == "__main__":
    main()