import time
import random

def create_list_matrix(size):
    return [[0 for _ in range(size)] for _ in range(size)]

def create_tuple_matrix(size):
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))

def test_list_performance(matrix, size, iterations):
    start_time = time.perf_counter()
    for _ in range(iterations):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        matrix[i][j] = i + j
    end_time = time.perf_counter()
    return end_time - start_time

def test_tuple_performance(matrix_wrapper, size, iterations):
    start_time = time.perf_counter()
    for _ in range(iterations):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        current_matrix = matrix_wrapper[0]
        original_row = current_matrix[i]
        new_row = original_row[:j] + (i + j,) + original_row[j+1:]
        new_matrix = current_matrix[:i] + (new_row,) + current_matrix[i+1:]
        matrix_wrapper[0] = new_matrix
    end_time = time.perf_counter()
    return end_time - start_time

def main():
    size = 10000
    iterations = 1000
    
    list_matrix = create_list_matrix(size)
    tuple_matrix_wrapper = [create_tuple_matrix(size)]
    
    list_time = test_list_performance(list_matrix, size, iterations)
    print(f"列表修改 {iterations} 次耗时: {list_time:.4f} 秒")
    
    tuple_time = test_tuple_performance(tuple_matrix_wrapper, size, iterations)
    print(f"元组修改 {iterations} 次耗时: {tuple_time:.4f} 秒")

if __name__ == "__main__":
    main()