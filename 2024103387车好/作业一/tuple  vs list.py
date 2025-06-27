import time
import sys


def test_container_performance():

    # 创建10000x10000的矩阵
    size = 10000  # 由于10000x10000会占用过多内存，这里调整为1000x1000
    rounds = 10000  # 修改轮数也相应减少

    # 创建list矩阵
    print("创建list矩阵...")
    list_matrix = [[0 for _ in range(size)] for _ in range(size)]

    # 创建tuple矩阵
    print("创建tuple矩阵...")
    tuple_matrix = tuple(tuple(0 for _ in range(size)) for _ in range(size))

    # 测试list修改性能
    print("测试list修改性能...")
    start_time = time.time()
    for _ in range(rounds):
        i, j = 0, 0  # 固定修改第一个元素以减少位置变化的影响
        # 需要创建一个新行来修改元素
        new_row = list(list_matrix[i])
        new_row[j] = 1
        list_matrix[i] = new_row
    list_duration = time.time() - start_time

    # 测试tuple修改性能
    print("测试tuple修改性能...")
    start_time = time.time()
    for _ in range(rounds):
        i, j = 0, 0  # 固定修改第一个元素以减少位置变化的影响
        # 需要重建整个元组
        new_matrix = []
        for row_idx, row in enumerate(tuple_matrix):
            if row_idx == i:
                new_row = list(row)
                new_row[j] = 1
                new_matrix.append(tuple(new_row))
            else:
                new_matrix.append(row)
        tuple_matrix = tuple(new_matrix)
    tuple_duration = time.time() - start_time

    print("\n性能测试结果:")
    print(f"List修改时间: {list_duration:.4f}秒")
    print(f"Tuple修改时间: {tuple_duration:.4f}秒")
    print(f"Tuple比List慢 {tuple_duration / list_duration:.1f}倍")

    # 内存占用测试
    print("\n内存占用测试:")
    list_size = sys.getsizeof(list_matrix) + sum(sys.getsizeof(row) for row in list_matrix)
    tuple_size = sys.getsizeof(tuple_matrix) + sum(sys.getsizeof(row) for row in tuple_matrix)
    print(f"List占用内存: {list_size / 1024 / 1024:.2f}MB")
    print(f"Tuple占用内存: {tuple_size / 1024 / 1024:.2f}MB")


if __name__ == "__main__":
    test_container_performance()
