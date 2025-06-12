import time
import random

MATRIX_SIZE = 10000
MODIFICATION_COUNT = 10000

def create_list_matrix(size):
    print(f"创建 {size}x{size} 列表矩阵...")
    start_time = time.time()
    matrix = [[random.randint(1, 100) for _ in range(size)] for _ in range(size)]
    creation_time = time.time() - start_time
    print(f"列表矩阵创建完成: {creation_time:.4f}秒")
    return matrix, creation_time

def create_tuple_matrix(size):
    print(f"创建 {size}x{size} 元组矩阵...")
    start_time = time.time()
    temp_matrix = [[random.randint(1, 100) for _ in range(size)] for _ in range(size)]
    matrix = tuple(tuple(row) for row in temp_matrix)
    creation_time = time.time() - start_time
    print(f"元组矩阵创建完成: {creation_time:.4f}秒")
    return matrix, creation_time

def modify_list_matrix(matrix, iterations):
    size = len(matrix)
    print(f"开始修改列表矩阵 {iterations} 次...")
    
    start_time = time.time()
    for i in range(iterations):
        row = random.randint(0, size-1)
        col = random.randint(0, size-1)
        matrix[row][col] = random.randint(1, 100)
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"进度: {i+1}/{iterations}, 用时: {elapsed:.2f}秒")
    
    modification_time = time.time() - start_time
    print(f"列表修改完成: {modification_time:.4f}秒")
    return modification_time

def modify_tuple_matrix(matrix, iterations):
    size = len(matrix)
    print(f"开始修改元组矩阵 {iterations} 次...")
    
    current_matrix = matrix
    start_time = time.time()
    
    for i in range(iterations):
        row = random.randint(0, size-1)
        col = random.randint(0, size-1)
        
        target_row_list = list(current_matrix[row])
        target_row_list[col] = random.randint(1, 100)
        new_row = tuple(target_row_list)
        matrix_list = list(current_matrix)
        matrix_list[row] = new_row
        current_matrix = tuple(matrix_list)
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"进度: {i+1}/{iterations}, 用时: {elapsed:.2f}秒")
    
    modification_time = time.time() - start_time
    print(f"元组修改完成: {modification_time:.4f}秒")
    return modification_time

def main():
    print("Python可变与不可变容器性能测试")
    print(f"矩阵: {MATRIX_SIZE}x{MATRIX_SIZE}, 修改: {MODIFICATION_COUNT}次")
    print("-" * 50)
    
    # 列表测试
    print("\n列表测试:")
    list_matrix, list_create_time = create_list_matrix(MATRIX_SIZE)
    list_modify_time = modify_list_matrix(list_matrix, MODIFICATION_COUNT)
    
    # 元组测试
    print("\n元组测试:")
    tuple_matrix, tuple_create_time = create_tuple_matrix(MATRIX_SIZE)
    tuple_modify_time = modify_tuple_matrix(tuple_matrix, MODIFICATION_COUNT)
    
    # 结果对比
    print("\n性能对比:")
    print(f"创建时间 - 列表: {list_create_time:.4f}秒, 元组: {tuple_create_time:.4f}秒")
    print(f"修改时间 - 列表: {list_modify_time:.4f}秒, 元组: {tuple_modify_time:.4f}秒")
    print(f"元组修改比列表慢 {tuple_modify_time/list_modify_time:.1f} 倍")

if __name__ == "__main__":
    main()