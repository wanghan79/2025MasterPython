import time
import random

def create_matrix(rows, cols, is_tuple=False):

    if is_tuple:
        return tuple(tuple(random.randint(0, 99) for _ in range(cols)) for _ in range(rows))
    else:
        return [[random.randint(0, 99) for _ in range(cols)] for _ in range(rows)]

def modify_list_matrix(matrix, num_modifications):

    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    start_time = time.time()
    for _ in range(num_modifications):
        row_idx = random.randint(0, rows - 1)
        col_idx = random.randint(0, cols - 1)
        matrix[row_idx][col_idx] = random.randint(0, 99)
    end_time = time.time()
    return end_time - start_time

def modify_tuple_matrix(matrix, num_modifications):

    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    
    start_time = time.time()
    current_matrix = list(list(row) for row in matrix) # 转换为可修改的列表形式以便操作
    
    for _ in range(num_modifications):
        row_idx = random.randint(0, rows - 1)
        col_idx = random.randint(0, cols - 1)
        new_value = random.randint(0, 99)

        temp_row = list(current_matrix[row_idx])
        temp_row[col_idx] = new_value

        temp_matrix = list(current_matrix)
        temp_matrix[row_idx] = temp_row
        current_matrix = temp_matrix # 更新当前矩阵，模拟元组的重建
        
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    MATRIX_SIZE = 10000
    MODIFICATIONS = 10000 

    print(f"开始测试，矩阵大小: {MATRIX_SIZE}x{MATRIX_SIZE}，修改次数: {MODIFICATIONS} 轮")

    print("\n创建列表矩阵...")
    list_matrix = create_matrix(MATRIX_SIZE, MATRIX_SIZE, is_tuple=False)
    print("列表矩阵创建完成。")

    print(f"开始修改列表矩阵（{MODIFICATIONS}轮，每轮修改一个元素）...")
    list_time = modify_list_matrix(list_matrix, MODIFICATIONS)
    print(f"列表矩阵修改完成，耗时: {list_time:.4f} 秒")

    print("\n创建元组矩阵...")

    tuple_matrix = create_matrix(MATRIX_SIZE, MATRIX_SIZE, is_tuple=True)
    print("元组矩阵创建完成。")

    print(f"开始修改元组矩阵（{MODIFICATIONS}轮，每轮修改一个元素）...")
    print("警告: 由于元组的不可变性，此操作将涉及大量对象创建，耗时将非常长，可能远超列表。")

    TUPLE_MODIFICATIONS = 10 
    print(f"为了避免长时间运行，元组矩阵的修改次数降为: {TUPLE_MODIFICATIONS} 轮")
    tuple_time = modify_tuple_matrix(tuple_matrix, TUPLE_MODIFICATIONS)
    print(f"元组矩阵修改完成，耗时: {tuple_time:.4f} 秒")

    print("\n--- 结果对比 ---")
    print(f"列表矩阵修改耗时: {list_time:.4f} 秒")
    print(f"元组矩阵修改耗时: {tuple_time:.4f} 秒 (请注意，元组修改次数已减少)")

    print("\n结论:")
    print("    - 列表（list）是可变的，可以直接修改元素，因此修改操作非常高效。")
    print("    - 元组（tuple）是不可变的，任何对元素的“修改”都意味着需要创建新的元组对象（至少是新的行元组和新的矩阵元组）。")
    print("      这导致元组在进行修改时性能开销巨大，通常远低于列表。")
    print("      在实际应用中，如果数据需要频繁修改，应优先考虑使用列表；元组适用于数据一旦创建就不再改变的场景。")

