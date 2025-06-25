import time

def test_container_modification(N=10000):
    print(f"正在构建 {N}x{N} 的 list 和 tuple ...")
    list_data = [list(range(N)) for _ in range(N)]
    tuple_data = tuple(tuple(range(N)) for _ in range(N))

    print("开始修改 list 数据...")
    start_list = time.time()
    for i in range(10000):
        row = i % N
        col = i % N
        list_data[row][col] = -1
    end_list = time.time()

    print("开始修改 tuple 数据...")
    start_tuple = time.time()
    for i in range(10000):
        row = i % N
        col = i % N
        temp = list(tuple_data[row])
        temp[col] = -1
        tuple_data = tuple_data[:row] + (tuple(temp),) + tuple_data[row+1:]
    end_tuple = time.time()

    print(f"list 修改耗时: {end_list - start_list:.4f} 秒")
    print(f"tuple 修改耗时: {end_tuple - start_tuple:.4f} 秒")

if __name__ == "__main__":
    test_container_modification()

