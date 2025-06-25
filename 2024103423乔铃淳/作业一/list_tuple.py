import time
import random
import sys

def test_list_and_tuple_performance():
    size = 10000  # 矩阵尺寸
    modifications = 10000  # 修改次数
    
    # 创建列表矩阵
    print("Creating list matrix...")
    list_start = time.time()
    list_matrix = [[0] * size for _ in range(size)]
    list_creation_time = time.time() - list_start
    print(f"List creation time: {list_creation_time:.4f} seconds")
    
    # 创建元组矩阵
    print("Creating tuple matrix...")
    tuple_start = time.time()
    tuple_matrix = tuple(tuple(row) for row in list_matrix)
    tuple_creation_time = time.time() - tuple_start
    print(f"Tuple creation time: {tuple_creation_time:.4f} seconds")
    
    # 生成随机修改位置
    random.seed(42)
    positions = [(random.randint(0, size-1), 
                 random.randint(0, size-1)) for _ in range(modifications)]
    
    # 测试列表修改性能
    print("\nTesting list modifications...")
    mod_start = time.time()
    for i, (row, col) in enumerate(positions):
        list_matrix[row][col] = i  # 直接修改元素
        
        # 进度显示
        if (i+1) % 1000 == 0:
            sys.stdout.write(f"\rProgress: {i+1}/{modifications}")
            sys.stdout.flush()
    
    list_mod_time = time.time() - mod_start
    print(f"\nList modification time: {list_mod_time:.4f} seconds")
    
    # 测试元组修改性能
    print("\nTesting tuple modifications...")
    current_tuple = tuple_matrix
    mod_start = time.time()
    
    for i, (row, col) in enumerate(positions):
        # 重建整个元组结构
        new_rows = []
        for r_idx, t_row in enumerate(current_tuple):
            if r_idx == row:
                # 重建被修改的行
                new_row = t_row[:col] + (i,) + t_row[col+1:]
                new_rows.append(new_row)
            else:
                new_rows.append(t_row)
        
        current_tuple = tuple(new_rows)  # 创建新元组
        
        # 进度显示
        if (i+1) % 100 == 0:  # 元组操作较慢，减少输出频率
            sys.stdout.write(f"\rProgress: {i+1}/{modifications}")
            sys.stdout.flush()
    
    tuple_mod_time = time.time() - mod_start
    print(f"\nTuple modification time: {tuple_mod_time:.4f} seconds")
    
    # 结果对比
    print("\n" + "="*50)
    print(f"List creation time:    {list_creation_time:>12.4f} seconds")
    print(f"Tuple creation time:   {tuple_creation_time:>12.4f} seconds")
    print(f"List modification time:  {list_mod_time:>12.4f} seconds")
    print(f"Tuple modification time: {tuple_mod_time:>12.4f} seconds")
    print(f"\nList is {tuple_mod_time/max(list_mod_time, 1e-9):.0f}x faster for modifications")
    
    return {
        'list_creation': list_creation_time,
        'tuple_creation': tuple_creation_time,
        'list_modification': list_mod_time,
        'tuple_modification': tuple_mod_time
    }

# 运行测试
if __name__ == "__main__":
    test_list_and_tuple_performance()