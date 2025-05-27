import time
import random

# 矩阵大小和修改次数
MATRIX_SIZE = 1000  # 使用较小的矩阵以避免内存问题
MODIFICATION_COUNT = 10000

def create_list_matrix(size):
    """创建一个二维列表矩阵"""
    return [[0 for _ in range(size)] for _ in range(size)]

def create_tuple_matrix(size):
    """创建一个二维元组矩阵"""
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))

def modify_list_matrix(matrix, iterations):
    """对列表矩阵进行多次随机位置的修改操作"""
    size = len(matrix)
    start_time = time.time()
    
    for _ in range(iterations):
        # 随机选择位置
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        # 修改值
        matrix[i][j] = random.randint(1, 100)
    
    end_time = time.time()
    return end_time - start_time

def modify_tuple_matrix(matrix, iterations):
    """对元组矩阵进行多次随机位置的修改操作"""
    size = len(matrix)
    start_time = time.time()
    
    for _ in range(iterations):
        # 随机选择位置
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        # 对于元组，需要创建新的元组来进行修改
        temp_row = list(matrix[i])
        temp_row[j] = random.randint(1, 100)
        matrix = matrix[:i] + (tuple(temp_row),) + matrix[i+1:]
    
    end_time = time.time()
    return end_time - start_time

def main():
    print(f"创建 {MATRIX_SIZE}x{MATRIX_SIZE} 大小的矩阵...")
    
    # 创建矩阵
    list_matrix = create_list_matrix(MATRIX_SIZE)
    tuple_matrix = create_tuple_matrix(MATRIX_SIZE)
    
    print(f"\n开始性能测试...")
    print(f"对每种矩阵类型执行 {MODIFICATION_COUNT} 次修改操作...")
    
    # 测试列表性能
    print("\n测试列表性能...")
    list_time = modify_list_matrix(list_matrix, MODIFICATION_COUNT)
    print(f"列表修改操作时间: {list_time:.6f} 秒")
    
    # 测试元组性能
    print("\n测试元组性能...")
    tuple_time = modify_tuple_matrix(tuple_matrix, MODIFICATION_COUNT)
    print(f"元组修改操作时间: {tuple_time:.6f} 秒")
    
    # 比较结果
    print("\n性能比较:")
    print(f"元组操作比列表操作慢 {tuple_time/list_time:.2f} 倍")
    
    # 分析结果
    print("\n性能差异分析:")
    print("列表是可变数据结构，可以直接修改其元素值，操作效率高。")
    print("元组是不可变数据结构，每次'修改'实际上是创建了新的元组对象，这需要额外的内存分配和数据复制，因此操作效率较低。")

if __name__ == "__main__":
    main()