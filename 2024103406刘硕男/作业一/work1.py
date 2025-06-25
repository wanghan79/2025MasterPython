import time
import numpy as np

def create_list_matrix(size):
    """创建并返回一个初始化为0的列表矩阵"""
    return [[0] * size for _ in range(size)]

def create_tuple_matrix(size):
    """创建并返回一个初始化为0的元组矩阵"""
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))

def modify_list_matrix(matrix, modifications):
    """修改列表矩阵指定位置的元素"""
    for i, j in modifications:
        matrix[i][j] = 1  # 直接修改元素

def modify_tuple_matrix(matrix, modifications, size):
    """通过重建方式'修改'元组矩阵（实际创建新元组）"""
    for idx, (i, j) in enumerate(modifications):
        # 将需要修改的行转换为列表
        row_list = list(matrix[i])
        row_list[j] = 1  # 修改该行指定位置
        
        # 重建整个矩阵（复用未修改的行）
        new_matrix = (
            matrix[:i] +                    # 保留前i行
            (tuple(row_list),) +            # 新修改的行
            matrix[i+1:]                   # 保留后面的行
        )
        matrix = new_matrix  # 更新矩阵引用
        
        # 每1000次输出进度（避免过多输出）
        if (idx + 1) % 1000 == 0:
            print(f"Tuple修改进度: {idx+1}/{len(modifications)}")
    return matrix

def main():
    SIZE = 10000     # 矩阵尺寸
    MODIFICATIONS = 10000  # 修改次数
    
    # 生成随机修改位置 (避免重复以模拟最坏情况)
    rng = np.random.default_rng()
    indices = rng.choice(SIZE * SIZE, size=MODIFICATIONS, replace=False)
    modifications = [(idx // SIZE, idx % SIZE) for idx in indices]
    
    # 测试列表矩阵修改
    print("创建列表矩阵...")
    list_matrix = create_list_matrix(SIZE)
    
    print("开始列表修改测试...")
    start_list = time.time()
    modify_list_matrix(list_matrix, modifications)
    end_list = time.time()
    list_time = end_list - start_list
    print(f"列表修改时间: {list_time:.4f}秒")
    
    # 测试元组矩阵修改
    print("\n创建元组矩阵...")
    tuple_matrix = create_tuple_matrix(SIZE)
    
    print("开始元组修改测试(可能需要较长时间)...")
    start_tuple = time.time()
    tuple_matrix = modify_tuple_matrix(tuple_matrix, modifications, SIZE)
    end_tuple = time.time()
    tuple_time = end_tuple - start_tuple
    print(f"元组修改时间: {tuple_time:.4f}秒")
    
    # 结果对比
    print("\n===== 测试结果对比 =====")
    print(f"列表修改总时间: {list_time:.4f}秒")
    print(f"元组修改总时间: {tuple_time:.4f}秒")
    print(f"元组操作比列表操作慢: {tuple_time / list_time:.1f}倍")

if __name__ == "__main__":
    main()
