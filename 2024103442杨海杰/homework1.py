import time
import random


def container_performance_test(matrix_size=1000, iterations=1000):

    print(f"创建 {matrix_size}×{matrix_size} 数据矩阵...")

    # 创建可变列表矩阵
    start = time.time()
    list_matrix = [[random.random() for _ in range(matrix_size)]
                   for _ in range(matrix_size)]
    list_creation_time = time.time() - start
    print(f"List 创建时间: {list_creation_time:.4f} 秒")

    # 创建不可变元组矩阵
    start = time.time()
    tuple_matrix = tuple(tuple(random.random() for _ in range(matrix_size))
                         for _ in range(matrix_size))
    tuple_creation_time = time.time() - start
    print(f"Tuple 创建时间: {tuple_creation_time:.4f} 秒")

    # 测试列表修改性能
    start = time.time()
    for _ in range(iterations):
        i = random.randint(0, matrix_size - 1)
        j = random.randint(0, matrix_size - 1)
        list_matrix[i][j] = random.random()
    list_mod_time = time.time() - start
    print(f"List 修改 {iterations} 个元素时间: {list_mod_time:.4f} 秒")

    # 测试元组修改性能 (实际是重建)
    start = time.time()
    mutable_matrix = list(map(list, tuple_matrix))

    for _ in range(iterations):
        i = random.randint(0, matrix_size - 1)
        j = random.randint(0, matrix_size - 1)
        mutable_matrix[i][j] = random.random()

    tuple_matrix = tuple(tuple(row) for row in mutable_matrix)
    tuple_mod_time = time.time() - start
    print(f"Tuple 重建 {iterations} 个元素时间: {tuple_mod_time:.4f} 秒")

    print("\n性能对比总结:")
    print(f"列表创建时间 ≈ {list_creation_time / tuple_creation_time:.2f}x 元组创建时间")
    print(f"列表修改时间 ≈ {list_mod_time / tuple_mod_time:.2f}x 元组重建时间")

    return {
        "list_creation": list_creation_time,
        "tuple_creation": tuple_creation_time,
        "list_modification": list_mod_time,
        "tuple_recreation": tuple_mod_time
    }


if __name__ == "__main__":
    # 使用较小尺寸避免内存问题，但保持高迭代次数以测试性能
    results = container_performance_test(matrix_size=1000, iterations=10000)