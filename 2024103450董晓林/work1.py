import time
import sys
from typing import List, Tuple, Union


def create_matrix(size: int, matrix_type: str = 'list') -> Union[List[List[int]], Tuple[Tuple[int, ...], ...]]:
    """创建指定大小的矩阵
    
    Args:
        size: 矩阵大小
        matrix_type: 矩阵类型 ('list' 或 'tuple')
    
    Returns:
        创建的矩阵
    """
    try:
        if matrix_type == 'list':
            return [[0 for _ in range(size)] for _ in range(size)]
        elif matrix_type == 'tuple':
            return tuple(tuple(0 for _ in range(size)) for _ in range(size))
        else:
            raise ValueError(f"不支持的矩阵类型: {matrix_type}")
    except MemoryError:
        print(f"错误：创建 {size}x{size} 的矩阵时内存不足")
        sys.exit(1)


def modify_matrix(matrix: Union[List[List[int]], Tuple[Tuple[int, ...], ...]], 
                 iterations: int) -> Union[List[List[int]], Tuple[Tuple[int, ...], ...]]:
    """修改矩阵元素
    
    Args:
        matrix: 要修改的矩阵
        iterations: 修改次数
    
    Returns:
        修改后的矩阵
    """
    size = len(matrix)
    is_tuple = isinstance(matrix, tuple)
    
    for i in range(iterations):
        # 使用简单的伪随机分布选择位置
        row = i % size
        col = (i * 7) % size
        
        if is_tuple:
            # 元组需要重建整个矩阵
            matrix = tuple(
                tuple(1 if (r == row and c == col) else matrix[r][c]
                      for c in range(size))
                for r in range(size)
            )
        else:
            # 列表可以直接修改
            new_row = list(matrix[row][:])
            new_row[col] = 1
            matrix[row] = new_row
            
    return matrix


def test_performance(size: int = 10000, iterations: int = 10000) -> None:
    """执行性能测试
    
    Args:
        size: 矩阵大小
        iterations: 测试迭代次数
    """
    print(f"\n=== 性能测试 ===")
    print(f"矩阵大小: {size}×{size}")
    print(f"迭代次数: {iterations}")
    
    # 测试列表性能
    print("\n测试列表性能...")
    start = time.time()
    list_matrix = create_matrix(size, 'list')
    list_matrix = modify_matrix(list_matrix, iterations)
    list_time = time.time() - start
    print(f"列表操作时间: {list_time:.4f}秒")
    
    # 测试元组性能
    print("\n测试元组性能...")
    start = time.time()
    tuple_matrix = create_matrix(size, 'tuple')
    tuple_matrix = modify_matrix(tuple_matrix, iterations)
    tuple_time = time.time() - start
    print(f"元组操作时间: {tuple_time:.4f}秒")
    
    # 输出性能比较
    print(f"\n性能比较:")
    print(f"元组操作比列表操作慢 {tuple_time / list_time:.1f} 倍")


if __name__ == "__main__":
    test_performance()