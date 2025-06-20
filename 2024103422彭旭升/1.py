import time
import random
import gc


def create_tuple_matrix(rows, cols):
    """创建tuple构成的矩阵"""
    print("正在创建tuple矩阵...")
    matrix = []
    for i in range(rows):
        row = tuple(random.randint(1, 100) for _ in range(cols))
        matrix.append(row)
    return tuple(matrix)


def create_list_matrix(rows, cols):
    """创建list构成的矩阵"""
    print("正在创建list矩阵...")
    matrix = []
    for i in range(rows):
        row = [random.randint(1, 100) for _ in range(cols)]
        matrix.append(row)
    return matrix


def modify_tuple_matrix(matrix, modifications):
    """修改tuple矩阵（需要重新创建）"""
    print("开始修改tuple矩阵...")
    matrix = list(matrix)  # 转换为可变结构

    for i in range(modifications):
        row_idx = random.randint(0, len(matrix) - 1)
        col_idx = random.randint(0, len(matrix[0]) - 1)

        # 将选中的行转为list进行修改
        row_list = list(matrix[row_idx])
        row_list[col_idx] = random.randint(1, 100)

        # 重新转换为tuple
        matrix[row_idx] = tuple(row_list)

        if (i + 1) % 1000 == 0:
            print(f"tuple修改进度: {i + 1}/{modifications}")

    return tuple(matrix)


def modify_list_matrix(matrix, modifications):
    """修改list矩阵"""
    print("开始修改list矩阵...")

    for i in range(modifications):
        row_idx = random.randint(0, len(matrix) - 1)
        col_idx = random.randint(0, len(matrix[0]) - 1)

        # 直接修改
        matrix[row_idx][col_idx] = random.randint(1, 100)

        if (i + 1) % 1000 == 0:
            print(f"list修改进度: {i + 1}/{modifications}")

    return matrix


def main():
    # 测试参数
    rows = 10000
    cols = 10000
    modifications = 10000

    print("=" * 60)
    print("Python容器性能测试")
    print(f"矩阵大小: {rows} × {cols}")
    print(f"修改次数: {modifications}")
    print("=" * 60)

    # 设置随机种子确保测试公平性
    random.seed(42)

    # 测试tuple性能
    print("\n【测试1: Tuple容器】")
    gc.collect()  # 清理内存

    start_time = time.time()
    tuple_matrix = create_tuple_matrix(rows, cols)
    create_tuple_time = time.time() - start_time
    print(f"tuple矩阵创建时间: {create_tuple_time:.2f}秒")

    start_time = time.time()
    modified_tuple_matrix = modify_tuple_matrix(tuple_matrix, modifications)
    modify_tuple_time = time.time() - start_time
    print(f"tuple矩阵修改时间: {modify_tuple_time:.2f}秒")

    total_tuple_time = create_tuple_time + modify_tuple_time
    print(f"tuple总时间: {total_tuple_time:.2f}秒")

    # 清理内存
    del tuple_matrix, modified_tuple_matrix
    gc.collect()

    # 重置随机种子
    random.seed(42)

    # 测试list性能
    print("\n【测试2: List容器】")
    gc.collect()  # 清理内存

    start_time = time.time()
    list_matrix = create_list_matrix(rows, cols)
    create_list_time = time.time() - start_time
    print(f"list矩阵创建时间: {create_list_time:.2f}秒")

    start_time = time.time()
    modified_list_matrix = modify_list_matrix(list_matrix, modifications)
    modify_list_time = time.time() - start_time
    print(f"list矩阵修改时间: {modify_list_time:.2f}秒")

    total_list_time = create_list_time + modify_list_time
    print(f"list总时间: {total_list_time:.2f}秒")

    # 结果对比
    print("\n" + "=" * 60)
    print("性能对比结果")
    print("=" * 60)
    print(f"创建时间对比:")
    print(f"  tuple: {create_tuple_time:.2f}秒")
    print(f"  list:  {create_list_time:.2f}秒")
    print(f"  差异:  {abs(create_tuple_time - create_list_time):.2f}秒")

    print(f"\n修改时间对比:")
    print(f"  tuple: {modify_tuple_time:.2f}秒")
    print(f"  list:  {modify_list_time:.2f}秒")
    print(f"  差异:  {abs(modify_tuple_time - modify_list_time):.2f}秒")

    print(f"\n总时间对比:")
    print(f"  tuple: {total_tuple_time:.2f}秒")
    print(f"  list:  {total_list_time:.2f}秒")
    print(f"  差异:  {abs(total_tuple_time - total_list_time):.2f}秒")

    # 性能分析
    if modify_tuple_time > modify_list_time:
        ratio = modify_tuple_time / modify_list_time
        print(f"\nlist修改性能比tuple快 {ratio:.1f} 倍")
    else:
        ratio = modify_list_time / modify_tuple_time
        print(f"\ntuple修改性能比list快 {ratio:.1f} 倍")

    print("\n分析结论:")
    print("- tuple是不可变对象，修改需要重新创建，开销很大")
    print("- list是可变对象，可以直接修改元素，效率更高")
    print("- 对于需要频繁修改的数据，应该使用list而不是tuple")


if __name__ == "__main__":
    main()