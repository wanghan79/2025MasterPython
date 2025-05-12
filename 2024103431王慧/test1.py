import time

def timed_operation(description, func):
    start = time.time()
    result = func()
    duration = time.time() - start
    print(f"{description:<25}: {duration:.6f} 秒")
    return result

# 创建 list 矩阵并修改其中一个值
list_matrix = timed_operation("创建 list 矩阵", lambda: [[0] * 10000 for _ in range(10000)])
timed_operation("修改 list 中一个值", lambda: list_matrix.__setitem__(5000, list_matrix[5000][:5000] + [1] + list_matrix[5000][5001:]))

# 创建 tuple 矩阵并修改其中一个值（需构造新 tuple）
tuple_matrix = timed_operation("创建 tuple 矩阵", lambda: tuple(tuple(0 for _ in range(10000)) for _ in range(10000)))

def modify_tuple_matrix(matrix):
    row = list(matrix[5000])
    row[5000] = 1
    modified_row = tuple(row)
    return tuple(
        modified_row if i == 5000 else matrix[i]
        for i in range(10000)
    )

timed_operation("修改 tuple 中一个值", lambda: modify_tuple_matrix(tuple_matrix))
