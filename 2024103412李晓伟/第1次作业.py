import time
import random

def create_tuple_matrix(rows, cols):
    #创建一个rows×cols的tuple矩阵
    return tuple(tuple(0 for _ in range(cols)) for _ in range(rows))

def create_list_matrix(rows, cols):
    #创建一个rows×cols的list矩阵
    return [[0 for _ in range(cols)] for _ in range(rows)]

def modify_tuple_matrix(matrix, iterations):

    rows = len(matrix)
    cols = len(matrix[0])
    start_time = time.time()

    for _ in range(iterations):
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)
        # 由于tuple不可变，每次修改都需要创建一个新的tuple
        new_row = tuple(v if idx != j else random.randint(1, 100) for idx, v in enumerate(matrix[i]))
        matrix = tuple(row if idx != i else new_row for idx, row in enumerate(matrix))

    end_time = time.time()
    return end_time - start_time, matrix

def modify_list_matrix(matrix, iterations):
    #直接修改list中的元素
    rows = len(matrix)
    cols = len(matrix[0])
    start_time = time.time()

    for _ in range(iterations):
        i = random.randint(0, rows - 1)
        j = random.randint(0, cols - 1)

        matrix[i][j] = random.randint(1, 100)

    end_time = time.time()
    return end_time - start_time, matrix


def main():
    # 设置矩阵大小和修改次数
    rows, cols = 10000, 10000
    iterations = 10000

    # 创建tuple矩阵
    print("\ntuple：")
    tuple_matrix = create_tuple_matrix(rows, cols)
    tuple_time, _ = modify_tuple_matrix(tuple_matrix, iterations)
    print(f"tuple 修改耗时: {tuple_time:.4f} 秒")

    # 创建list矩阵
    print("\nlist：")
    list_matrix = create_list_matrix(rows, cols)
    list_time, _ = modify_list_matrix(list_matrix, iterations)
    print(f"list 修改耗时: {list_time:.4f} 秒")

if __name__ == "__main__":
    main()

# 输出：
# tuple：
# tuple 修改耗时: 33.0488 秒
#
# list：
# list 修改耗时: 0.0171 秒

