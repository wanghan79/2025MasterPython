import time
import numpy as np

np.random.seed(42)

def create_tuple_matrix(rows, cols):
    return tuple(tuple(row) for row in np.random.randint(0, 100, size=(rows, cols)))

def create_list_matrix(rows, cols):
    return np.random.randint(0, 100, size=(rows, cols)).tolist()

def modify_tuple_matrix(matrix, rounds):
    rows = len(matrix)
    cols = len(matrix[0])
    current_matrix = matrix
    
    for _ in range(rounds):
        i = np.random.randint(0, rows)
        j = np.random.randint(0, cols)
        new_value = np.random.randint(0, 100)
        new_row = list(current_matrix[i])
        new_row[j] = new_value
        new_matrix = list(current_matrix)
        new_matrix[i] = tuple(new_row)
        current_matrix = tuple(new_matrix)
    
    return current_matrix

def modify_list_matrix(matrix, rounds):
    rows = len(matrix)
    cols = len(matrix[0])
    
    for _ in range(rounds):
        i = np.random.randint(0, rows)
        j = np.random.randint(0, cols)
        new_value = np.random.randint(0, 100)
    
        matrix[i][j] = new_value
    
    return matrix

def run_performance_test(rows, cols, rounds):
    print(f"测试配置: {rows}×{cols} 矩阵，{rounds}轮修改")
    print("\n测试tuple性能...")
    start_time = time.time()
    tuple_matrix = create_tuple_matrix(rows, cols)
    create_tuple_time = time.time() - start_time
    
    start_time = time.time()
    modify_tuple_matrix(tuple_matrix, rounds)
    modify_tuple_time = time.time() - start_time
   
    print(f"tuple创建耗时: {create_tuple_time:.4f}秒")
    print(f"tuple修改耗时: {modify_tuple_time:.4f}秒")

    print("\n测试list性能...")
    start_time = time.time()
    list_matrix = create_list_matrix(rows, cols)
    create_list_time = time.time() - start_time
    
    start_time = time.time()
    modify_list_matrix(list_matrix, rounds)
    modify_list_time = time.time() - start_time
    
    print(f"list创建耗时: {create_list_time:.4f}秒")
    print(f"list修改耗时: {modify_list_time:.4f}秒")

    creation_ratio = create_tuple_time / create_list_time
    modification_ratio = modify_tuple_time / modify_list_time
    
    print("\n性能对比:")
    print(f"创建速度比(tuple/list): {creation_ratio:.2f}x")
    print(f"修改速度比(tuple/list): {modification_ratio:.2f}x")

if __name__ == "__main__":
    ROWS = 10000
    COLS = 10000
    ROUNDS = 10000
    run_performance_test(ROWS, COLS, ROUNDS)