import time
import random
import copy

def modify_list_matrix(matrix, num_modifications=10000):

    start = time.time()
    for _ in range(num_modifications):
        i = random.randint(0, len(matrix) - 1)
        j = random.randint(0, len(matrix[0]) - 1)
        matrix[i][j] = matrix[i][j] + 1  # 简单修改
    end = time.time()
    return end - start

def modify_tuple_matrix(matrix, num_modifications=10000):

    start = time.time()
    for _ in range(num_modifications):
        i = random.randint(0, len(matrix) - 1)
        j = random.randint(0, len(matrix[0]) - 1)
        # tuple 不可变，需先转 list 修改再转回去
        row = list(matrix[i])
        row[j] = row[j] + 1
        matrix[i] = tuple(row)
    end = time.time()
    return end - start

def main():
    size = 10000
    num_modifications = 10000

    print("构造 list 矩阵...")
    list_matrix = [[0] * size for _ in range(size)]

    print("构造 tuple 矩阵...")
    tuple_matrix = tuple(tuple(0 for _ in range(size)) for _ in range(size))

    print("复制 list 和 tuple 矩阵用于修改...")
    list_matrix_copy = copy.deepcopy(list_matrix)
    tuple_matrix_copy = list(tuple(row) for row in list_matrix)  # 保证结构一致

    print("开始修改 list 矩阵...")
    list_time = modify_list_matrix(list_matrix_copy, num_modifications)
    print(f"list 修改耗时: {list_time:.2f} 秒")

    print("开始修改 tuple 矩阵...")
    tuple_time = modify_tuple_matrix(list(tuple_matrix_copy), num_modifications)
    print(f"tuple 修改耗时: {tuple_time:.2f} 秒")

if __name__ == "__main__":
    main()
