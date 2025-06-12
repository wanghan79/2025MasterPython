import time
import copy

def test_list_modification():
    # 创建10000×10000的list矩阵
    print("Creating 10000x10000 list matrix...")
    start_time = time.time()
    my_list = [[0 for _ in range(10000)] for _ in range(10000)]
    creation_time = time.time() - start_time
    print(f"List creation time: {creation_time:.4f} seconds")
    
    # 测试10000次修改
    print("Testing 10000 modifications...")
    start_time = time.time()
    for i in range(10000):
        # 修改不同位置的元素以避免缓存优化
        my_list[i % 10000][i % 10000] = 1
    modification_time = time.time() - start_time
    print(f"List modification time: {modification_time:.4f} seconds")
    print(f"Average per modification: {modification_time/10000:.6f} seconds\n")
    return creation_time, modification_time

def test_tuple_modification():
    # 创建10000×10000的tuple矩阵
    print("Creating 10000x10000 tuple matrix...")
    start_time = time.time()
    my_tuple = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))
    creation_time = time.time() - start_time
    print(f"Tuple creation time: {creation_time:.4f} seconds")
    
    # 测试10000次"修改"（实际上是重建）
    print("Testing 10000 modifications (tuple recreation)...")
    start_time = time.time()
    temp_tuple = my_tuple
    for i in range(10000):
        # 将tuple转换为list进行修改
        temp_list = [list(row) for row in temp_tuple]
        # 修改元素
        temp_list[i % 10000][i % 10000] = 1
        # 转换回tuple
        temp_tuple = tuple(tuple(row) for row in temp_list)
    modification_time = time.time() - start_time
    print(f"Tuple modification (recreation) time: {modification_time:.4f} seconds")
    print(f"Average per modification: {modification_time/10000:.6f} seconds\n")
    return creation_time, modification_time

def main():
    print("=== Python可变与不可变数据结构性能测试 ===")
    print("=== 测试list和tuple的修改操作性能差异 ===\n")
    
    # 测试list性能
    list_creation, list_mod = test_list_modification()
    
    # 测试tuple性能
    tuple_creation, tuple_mod = test_tuple_modification()
    
    # 性能比较
    print("\n=== 性能比较结果 ===")
    print(f"创建时间比(tuple/list): {tuple_creation/list_creation:.2f}x")
    print(f"修改时间比(tuple/list): {tuple_mod/list_mod:.2f}x")
    
    print("\n=== 结论 ===")
    print("1. 列表(list)是可变数据结构，可以直接修改元素，速度非常快")
    print("2. 元组(tuple)是不可变数据结构，'修改'操作需要重建整个数据结构，性能极差")
    print("3. 在需要频繁修改数据的场景中，应该使用list而不是tuple")

if __name__ == "__main__":
    main()