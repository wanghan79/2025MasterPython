import time
import numpy as np

def test_list_performance():
    # 创建10000x10000的列表矩阵
    matrix = [[0 for _ in range(10000)] for _ in range(10000)]
    
    start_time = time.time()
    
    # 进行10000次修改
    for i in range(10000):
        row = np.random.randint(0, 10000)
        col = np.random.randint(0, 10000)
        matrix[row][col] = 1
    
    end_time = time.time()
    return end_time - start_time

def test_tuple_performance():
    # 创建10000x10000的元组矩阵
    matrix = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))
    
    start_time = time.time()
    
    # 进行10000次修改
    for i in range(10000):
        row = np.random.randint(0, 10000)
        col = np.random.randint(0, 10000)
        # 由于元组是不可变的，我们需要重建整个矩阵
        matrix = tuple(
            tuple(1 if r == row and c == col else matrix[r][c] 
            for c in range(10000))
            for r in range(10000)
        )
    
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    print("开始性能测试...")
    
    list_time = test_list_performance()
    print(f"列表修改耗时: {list_time:.2f} 秒")
    
    tuple_time = test_tuple_performance()
    print(f"元组修改耗时: {tuple_time:.2f} 秒")
    
    print(f"性能差异倍数: {tuple_time/list_time:.2f} 倍") 