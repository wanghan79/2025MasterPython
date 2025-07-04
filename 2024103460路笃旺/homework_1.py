import time
import random


def create_test_data():
    list_matrix = [[0] * 10000 for _ in range(10000)]
    tuple_matrix = [tuple(0 for _ in range(10000)) for _ in range(10000)]
    positions = [(random.randint(0, 9999), random.randint(0, 9999)) for _ in range(10000)]
    return list_matrix, tuple_matrix, positions


def test_list_modification(list_matrix, positions):
    start_time = time.time()
    for i, j in positions:
        list_matrix[i][j] = 1
    return time.time() - start_time


def test_tuple_modification(tuple_matrix, positions):
    start_time = time.time()
    for i, j in positions:
        row_list = list(tuple_matrix[i])
        row_list[j] = 1
        tuple_matrix[i] = tuple(row_list)
    return time.time() - start_time


def main():
    print("正在创建10000×10000测试矩阵...")
    list_mat, tuple_mat, rand_positions = create_test_data()

    print("\n测试list矩阵10000次单元素修改...")
    list_time = test_list_modification(list_mat, rand_positions)
    print(f"List修改耗时: {list_time:.4f}秒")

    print("\n测试tuple矩阵10000次单元素修改...")
    tuple_time = test_tuple_modification(tuple_mat, rand_positions)
    print(f"Tuple修改耗时: {tuple_time:.4f}秒")

    print("\n性能对比:")
    print(f"List操作速度是Tuple的 {tuple_time / list_time:.1f} 倍")
    print(f"Tuple操作比List慢 {tuple_time - list_time:.4f} 秒")


if __name__ == "__main__":
    main()