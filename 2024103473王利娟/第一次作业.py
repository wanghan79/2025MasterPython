import time
import random

def create_list_matrix(size):
    """创建列表矩阵"""
    return [[0 for _ in range(size)] for _ in range(size)]

def create_tuple_matrix(size):
    """创建元组矩阵"""
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))

def modify_list_matrix(matrix, modifications):
    """修改列表矩阵"""
    for _ in range(modifications):
        i = random.randint(0, len(matrix) - 1)
        j = random.randint(0, len(matrix[0]) - 1)
        matrix[i][j] = random.randint(0, 100)

def modify_tuple_matrix(matrix, modifications):
    """修改元组矩阵"""
    temp_matrix = [list(row) for row in matrix]  # 转换为列表便于修改
    for _ in range(modifications):
        i = random.randint(0, len(temp_matrix) - 1)
        j = random.randint(0, len(temp_matrix[0]) - 1)
        temp_matrix[i][j] = random.randint(0, 100)
    return tuple(tuple(row) for row in temp_matrix)  # 转换回元组

def main():
    size = 10000
    modifications = 10000

    # 创建矩阵
    list_matrix = create_list_matrix(size)
    tuple_matrix = create_tuple_matrix(size)

    # 测试列表修改性能
    start_list = time.time()
    modify_list_matrix(list_matrix, modifications)
    end_list = time.time()

    # 测试元组修改性能
    start_tuple = time.time()
    tuple_matrix = modify_tuple_matrix(tuple_matrix, modifications)
    end_tuple = time.time()

    # 输出结果
    print(f"\n列表矩阵修改时间: {end_list - start_list:.4f} 秒")
    print(f"元组矩阵修改时间: {end_tuple - start_tuple:.4f} 秒")


if __name__ == "__main__":
    main()