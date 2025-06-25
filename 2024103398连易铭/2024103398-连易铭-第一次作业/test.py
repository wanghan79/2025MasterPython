import time

# 创建一个可变的矩阵，就是用列表
def build_list_matrix(n):
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        matrix.append(row)
    return matrix

# 创建一个不可变的矩阵，就是用元组
def build_tuple_matrix(n):
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        matrix.append(tuple(row))
    return tuple(matrix)

# 修改列表矩阵，在里面某些位置放1
def change_list_matrix(matrix, times):
    length = len(matrix)
    for i in range(times):
        r = i % length
        c = (i * 7) % length
        new_line = []
        for value in matrix[r]:
            new_line.append(value)
        new_line[c] = 1
        matrix[r] = new_line

# 修改元组矩阵（其实是重做一份）
def change_tuple_matrix(matrix, times):
    length = len(matrix)
    for i in range(times):
        r = i % length
        c = (i * 7) % length
        new_matrix = []
        for row_index in range(length):
            new_row = []
            for col_index in range(length):
                if row_index == r and col_index == c:
                    new_row.append(1)
                else:
                    new_row.append(matrix[row_index][col_index])
            new_matrix.append(tuple(new_row))
        matrix = tuple(new_matrix)

# 主程序，测试性能
def main():
    size = 10000
    times = 10000
    print("开始测试")
    print("矩阵大小是", size, "，修改次数是", times)

    start1 = time.time()
    m1 = build_list_matrix(size)
    change_list_matrix(m1, times)
    end1 = time.time()
    print("用列表修改用了", round(end1 - start1, 4), "秒")

    start2 = time.time()
    m2 = build_tuple_matrix(size)
    change_tuple_matrix(m2, times)
    end2 = time.time()
    print("用元组修改用了", round(end2 - start2, 4), "秒")

    print("元组修改是列表的", round((end2 - start2) / (end1 - start1), 1), "倍慢")

# 只有在直接运行时才执行
if __name__ == "__main__":
    main()
