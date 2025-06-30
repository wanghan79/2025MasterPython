import time
import numpy as np

def test_list_modification():
    # 创建一个10000x10000的列表
    print("创建列表矩阵...")
    list_matrix = [[0 for _ in range(10000)] for _ in range(10000)]
    
    # 记录开始时间
    start_time = time.time()
    
    # 进行10000轮修改，每轮修改一个元素
    print("开始列表修改测试...")
    for i in range(10000):
        # 随机选择一个位置进行修改
        row = np.random.randint(0, 10000)
        col = np.random.randint(0, 10000)
        list_matrix[row][col] = i
    
    # 记录结束时间
    end_time = time.time()
    
    # 计算总耗时
    elapsed_time = end_time - start_time
    print(f"列表修改10000次耗时: {elapsed_time:.6f} 秒")
    
    return elapsed_time

def test_tuple_modification():
    # 创建一个10000x10000的元组矩阵
    # 注意：我们不能直接创建一个这么大的元组矩阵，因为内存会爆炸
    # 所以我们创建一个小一点的矩阵用于测试
    print("创建元组矩阵...")
    tuple_size = 100  # 使用较小的尺寸，因为tuple修改成本极高
    tuple_matrix = tuple(tuple(0 for _ in range(tuple_size)) for _ in range(tuple_size))
    
    # 记录开始时间
    start_time = time.time()
    
    # 进行10000轮修改，每轮修改一个元素
    print("开始元组修改测试...")
    for i in range(10000):
        # 随机选择一个位置进行修改
        row = np.random.randint(0, tuple_size)
        col = np.random.randint(0, tuple_size)
        
        # 元组是不可变的，必须创建一个新的元组
        # 首先将要修改的行转换为列表
        temp_row = list(tuple_matrix[row])
        # 修改列表中的元素
        temp_row[col] = i
        # 将修改后的列表转换回元组
        new_row = tuple(temp_row)
        
        # 将整个tuple_matrix转换为列表以便修改
        temp_matrix = list(tuple_matrix)
        # 替换修改后的行
        temp_matrix[row] = new_row
        # 将修改后的列表转换回元组
        tuple_matrix = tuple(temp_matrix)
    
    # 记录结束时间
    end_time = time.time()
    
    # 计算总耗时
    elapsed_time = end_time - start_time
    print(f"元组修改10000次耗时: {elapsed_time:.6f} 秒")
    
    return elapsed_time

def main():
    print("开始测试列表和元组的性能差异...")
    
    # 测试列表修改性能
    list_time = test_list_modification()
    
    # 测试元组修改性能
    tuple_time = test_tuple_modification()
    
    # 对比结果
    print("\n性能对比结果:")
    print(f"列表修改10000次耗时: {list_time:.6f} 秒")
    print(f"元组修改10000次耗时: {tuple_time:.6f} 秒")
    print(f"元组修改时间是列表的 {tuple_time/list_time:.2f} 倍")
    
    print("\n结论:")
    print("1. 列表是可变的，可以直接修改某个元素，速度很快。")
    print("2. 元组是不可变的，每次修改都要重建整个元组，导致性能极差。")
    print("3. 对于需要频繁修改的数据，应当使用列表而非元组。")

if __name__ == "__main__":
    main() 