import time
import sys


def create_list_matrix(size):
    """创建列表矩阵（可变）"""
    return [[0 for _ in range(size)] for _ in range(size)]


def create_tuple_matrix(size):
    """创建元组矩阵（不可变）"""
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))


def modify_list_matrix(matrix, iterations):
    """修改列表矩阵"""
    size = len(matrix)
    for i in range(iterations):
        # 随机选择一个位置修改（这里简化使用循环索引）
        row = i % size
        col = (i * 7) % size  # 简单的伪随机分布
        # 需要创建新行来"修改"元素
        new_row = list(matrix[row][:])
        new_row[col] = 1
        matrix[row] = new_row


def modify_tuple_matrix(matrix, iterations):
    """修改元组矩阵 - 实际上需要完全重建"""
    size = len(matrix)
    for i in range(iterations):
        row = i % size
        col = (i * 7) % size
        # 必须重建整个矩阵来修改一个元素
        matrix = tuple(
            tuple(1 if (r == row and c == col) else matrix[r][c]
                  for c in range(size))
            for r in range(size)
        )


def test_performance():
    size = 10000
    iterations = 10000

    print(f"测试矩阵大小: {size}×{size}, 迭代次数: {iterations}")

    # 测试列表性能
    start = time.time()
    list_matrix = create_list_matrix(size)
    modify_list_matrix(list_matrix, iterations)
    list_time = time.time() - start
    print(f"列表操作时间: {list_time:.4f}秒")

    # 测试元组性能
    start = time.time()
    tuple_matrix = create_tuple_matrix(size)
    modify_tuple_matrix(tuple_matrix, iterations)
    tuple_time = time.time() - start
    print(f"元组操作时间: {tuple_time:.4f}秒")

    print(f"元组操作比列表操作慢 {tuple_time / list_time:.1f} 倍")


if __name__ == "__main__":
    test_performance()