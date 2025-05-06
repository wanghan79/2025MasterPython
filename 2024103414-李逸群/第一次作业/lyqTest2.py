
import time
import random



def create_list_matrix(size):

    return [[0] * size for _ in range(size)]


def create_tuple_matrix(size):

    return tuple(tuple([0] * size) for _ in range(size))


def modify_list_matrix(matrix, iterations, size):


    start_time = time.time()
    for _ in range(iterations):
        row = random.randint(0, size - 1)
        col = random.randint(0, size - 1)
        matrix[row][col] = 1
    end_time = time.time()
    return end_time - start_time


def modify_tuple_matrix(matrix, iterations, size):

    start_time = time.time()
    for _ in range(iterations):
        row = random.randint(0, size - 1)
        col = random.randint(0, size - 1)
        new_row = list(matrix[row])
        new_row[col] = 1
        new_matrix = list(matrix)
        new_matrix[row] = tuple(new_row)
        matrix = tuple(new_matrix)
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    size = 10000
    iterations = 10000

    # 创建列表矩阵和元组矩阵
    list_matrix = create_list_matrix(size)
    tuple_matrix = create_tuple_matrix(size)

    # 测试列表矩阵的修改性能
    list_time = modify_list_matrix(list_matrix, iterations, size)

    # 测试元组矩阵的修改性能
    tuple_time = modify_tuple_matrix(tuple_matrix, iterations, size)

    # 输出结果
    print(f"使用 list 修改矩阵花费的时间: {list_time} 秒")
    print(f"使用 tuple 修改矩阵花费的时间: {tuple_time} 秒")