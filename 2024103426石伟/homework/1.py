mport time
import random


def test_list_modification():
    # 创建10000×10000的列表矩阵（可变）
    matrix = [[0] * 10000 for _ in range(10000)]

    start_time = time.time()
    # 进行10000轮修改
    for _ in range(10000):
        i = random.randint(0, 9999)
        j = random.randint(0, 9999)
        matrix[i][j] = 1  # 直接修改元素
    return time.time() - start_time


def test_tuple_modification():
    # 创建10000×10000的元组矩阵（不可变）
    matrix = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))

    start_time = time.time()
    # 进行10000轮修改
    for _ in range(10000):
        i = random.randint(0, 9999)
        j = random.randint(0, 9999)
        # 重建整个矩阵（每次修改需创建新元组）
        new_matrix = tuple(
            tuple(1 if (x == i and y == j) else matrix[x][y]
                  for y in range(10000))
            for x in range(10000)
        )
        matrix = new_matrix  # 更新矩阵引用
    return time.time() - start_time


# 执行测试
print("测试列表修改...")
list_time = test_list_modification()
print(f"列表修改耗时: {list_time:.4f} 秒")

print("\n测试元组修改...")
tuple_time = test_tuple_modification()
print(f"元组修改耗时: {tuple_time:.4f} 秒")

# 性能对比
print("\n性能对比结果:")
print(f"列表速度是元组的 {tuple_time / list_time:.1f} 倍")
