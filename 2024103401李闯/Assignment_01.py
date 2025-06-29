import time
import random


def test_list_modification():
    # 创建10000×10000列表矩阵
    matrix = [[0] * 10000 for _ in range(10000)]

    # 进行10000次随机修改
    start_time = time.time()
    for _ in range(10000):
        i = random.randint(0, 9999)
        j = random.randint(0, 9999)
        matrix[i][j] = 1  # 直接修改元素
    return time.time() - start_time


def test_tuple_modification():
    # 创建10000×10000元组矩阵
    matrix = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))

    # 进行10000次随机修改
    start_time = time.time()
    for _ in range(10000):
        i = random.randint(0, 9999)
        j = random.randint(0, 9999)

        # 重建整个矩阵来"修改"一个元素
        new_matrix = []
        for row_idx, row in enumerate(matrix):
            if row_idx == i:
                new_row = list(row)
                new_row[j] = 1
                new_matrix.append(tuple(new_row))
            else:
                new_matrix.append(row)
        matrix = tuple(new_matrix)
    return time.time() - start_time


# 测试列表修改性能
list_time = test_list_modification()
print(f"List modification time: {list_time:.4f} seconds")

# 测试元组修改性能
tuple_time = test_tuple_modification()
print(f"Tuple modification time: {tuple_time:.4f} seconds")

# 性能对比
print(f"\nPerformance comparison:")
print(f"List is {tuple_time / list_time:.1f}x faster than tuple")