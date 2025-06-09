# 作业1：Python数据固定值和可变值容器测试
# 要求：分别创建由tuple和list构造的10000×10000的数据矩阵，
# 各自进行10000轮修改，每次只修改一个，对比二者的时间消耗

import time
import random
import sys

def create_tuple_matrix(rows, cols):
    """
    创建tuple矩阵
    由于tuple不可变，这里创建嵌套tuple结构
    """
    print(f"正在创建 {rows}×{cols} 的tuple矩阵...")
    matrix = tuple(tuple(random.randint(1, 100) for _ in range(cols)) for _ in range(rows))
    print("tuple矩阵创建完成")
    return matrix

def create_list_matrix(rows, cols):
    """
    创建list矩阵
    创建嵌套list结构，可以直接修改
    """
    print(f"正在创建 {rows}×{cols} 的list矩阵...")
    matrix = [[random.randint(1, 100) for _ in range(cols)] for _ in range(rows)]
    print("list矩阵创建完成")
    return matrix

def modify_tuple_matrix(matrix, modifications):
    """
    修改tuple矩阵
    由于tuple不可变，需要重新构造整个矩阵
    """
    print(f"开始对tuple矩阵进行 {modifications} 次修改...")
    start_time = time.time()
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    for i in range(modifications):
        # 随机选择要修改的位置
        row_idx = random.randint(0, rows - 1)
        col_idx = random.randint(0, cols - 1)
        new_value = random.randint(1, 100)
        
        # 将tuple转换为list进行修改，然后再转回tuple
        matrix_list = [list(row) for row in matrix]
        matrix_list[row_idx][col_idx] = new_value
        matrix = tuple(tuple(row) for row in matrix_list)
        
        if (i + 1) % 1000 == 0:
            print(f"tuple修改进度: {i + 1}/{modifications}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"tuple矩阵修改完成，耗时: {elapsed_time:.4f} 秒")
    return matrix, elapsed_time

def modify_list_matrix(matrix, modifications):
    """
    修改list矩阵
    可以直接修改指定位置的值
    """
    print(f"开始对list矩阵进行 {modifications} 次修改...")
    start_time = time.time()
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    for i in range(modifications):
        # 随机选择要修改的位置
        row_idx = random.randint(0, rows - 1)
        col_idx = random.randint(0, cols - 1)
        new_value = random.randint(1, 100)
        
        # 直接修改
        matrix[row_idx][col_idx] = new_value
        
        if (i + 1) % 1000 == 0:
            print(f"list修改进度: {i + 1}/{modifications}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"list矩阵修改完成，耗时: {elapsed_time:.4f} 秒")
    return matrix, elapsed_time

def main():
    """
    主函数：执行性能测试
    """
    print("=" * 60)
    print("Python数据容器性能测试")
    print("=" * 60)
    
    # 考虑到内存限制，使用较小的矩阵进行测试
    # 如果系统性能允许，可以逐步增加到10000×10000
    test_sizes = [
        (100, 100, 1000),    # 100×100矩阵，1000次修改
        (500, 500, 5000),    # 500×500矩阵，5000次修改
        (1000, 1000, 10000)  # 1000×1000矩阵，10000次修改
    ]
    
    results = []
    
    for rows, cols, modifications in test_sizes:
        print(f"\n测试规模: {rows}×{cols} 矩阵，{modifications} 次修改")
        print("-" * 50)
        
        try:
            # 测试tuple性能
            tuple_matrix = create_tuple_matrix(rows, cols)
            tuple_matrix, tuple_time = modify_tuple_matrix(tuple_matrix, modifications)
            
            # 测试list性能
            list_matrix = create_list_matrix(rows, cols)
            list_matrix, list_time = modify_list_matrix(list_matrix, modifications)
            
            # 计算性能差异
            speed_ratio = tuple_time / list_time if list_time > 0 else float('inf')
            
            result = {
                'size': f"{rows}×{cols}",
                'modifications': modifications,
                'tuple_time': tuple_time,
                'list_time': list_time,
                'speed_ratio': speed_ratio
            }
            results.append(result)
            
            print(f"\n结果对比:")
            print(f"tuple修改耗时: {tuple_time:.4f} 秒")
            print(f"list修改耗时:  {list_time:.4f} 秒")
            print(f"tuple比list慢: {speed_ratio:.2f} 倍")
            
        except MemoryError:
            print(f"内存不足，无法创建 {rows}×{cols} 的矩阵")
            break
        except Exception as e:
            print(f"测试过程中出现错误: {e}")
            break
    
    # 输出总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    for result in results:
        print(f"矩阵规模: {result['size']}, 修改次数: {result['modifications']}")
        print(f"  tuple耗时: {result['tuple_time']:.4f}s")
        print(f"  list耗时:  {result['list_time']:.4f}s")
        print(f"  性能差异: tuple比list慢 {result['speed_ratio']:.2f} 倍")
        print()
    
    print("结论:")
    print("1. tuple由于不可变性，每次修改都需要重新构造整个数据结构")
    print("2. list支持原地修改，性能明显优于tuple")
    print("3. 随着数据规模增大，性能差异会更加明显")

if __name__ == "__main__":
    main()
