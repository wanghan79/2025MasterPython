import time
import random

def create_list_matrix(rows=10000, cols=10000):
    """创建10000x10000的列表矩阵"""
    return [[random.randint(0, 100) for _ in range(cols)] for _ in range(rows)]

def create_tuple_matrix(rows=10000, cols=10000):
    """创建10000x10000的元组矩阵"""
    return tuple(tuple(random.randint(0, 100) for _ in range(cols)) for _ in range(rows))


def test_list_modification(matrix):
    """测试列表矩阵的修改时间"""
    start = time.time()
    for _ in range(10000):
        i = random.randint(0, 9999)
        j = random.randint(0, 9999)
        matrix[i][j] = random.randint(0, 100)
    return time.time() - start

def test_tuple_modification(matrix):
    """测试元组矩阵的修改时间"""
    start = time.time()
    for _ in range(10000):
        i = random.randint(0, 9999)
        j = random.randint(0, 9999)
        new_val = random.randint(0, 100)
        # 元组不可变，需重构整行
        row = list(matrix[i])
        row[j] = new_val
        new_row = tuple(row)
        matrix = matrix[:i] + (new_row,) + matrix[i + 1:]
    return time.time() - start

if __name__ == "__main__":
    # 初始化矩阵
    list_mat = create_list_matrix()
    tuple_mat = create_tuple_matrix()
    # 测试列表修改时间
    list_time = test_list_modification(list_mat)
    # 测试元组修改时间
    tuple_time = test_tuple_modification(tuple_mat)

    print(f"List 10000次修改耗时: {list_time:.4f} 秒")
    print(f"Tuple 10000次修改耗时: {tuple_time:.4f} 秒")