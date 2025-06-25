import time
import random

SIZE = 10000
ROUNDS = 10000

# 使用 list 进行实验
list_array = [[0] * SIZE for _ in range(SIZE)]

start_time = time.time()
for _ in range(ROUNDS):
    i, j = random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)
    list_array[i][j] = random.randint(1, 100)
list_time = time.time() - start_time
print(f"List 修改耗时: {list_time:.2f} 秒")

# 使用 tuple 进行实验（会非常慢）
tuple_array = tuple(tuple(0 for _ in range(SIZE)) for _ in range(SIZE))

start_time = time.time()
for _ in range(ROUNDS):
    i, j = random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)
    row = tuple_array[i][:j] + (random.randint(1, 100),) + tuple_array[i][j + 1:]  # 修改第 i 行的 j 列
    tuple_array = tuple_array[:i] + (row,) + tuple_array[i + 1:]  # 替换整个 i 行
tuple_time = time.time() - start_time
print(f"Tuple 修改耗时: {tuple_time:.2f} 秒")


# 使用 tuple 存储学生信息
# student1 = (1001, "张三", "男", "zhangsan@example.com", "12345678901", {"数学": "A", "英语": "B+"})
# student2 = (1002, "李四", "女", "lisi@example.com", "10987654321", {"物理": "A-", "化学": "B"})

# 存入 tuple 变量
# students_tuple = (student1, student2)
#
# # 打印信息
# for student in students_tuple:
#     print(f"学号: {student[0]}, 姓名: {student[1]}, 性别: {student[2]}, 邮箱: {student[3]}, 电话: {student[4]}, 选课信息: {student[5]}")


# 使用 list 存储学生信息
# students_list = [
#     [1001, "张三", "男", "zhangsan@example.com", "12345678901", {"数学": "A", "英语": "B+"}],
#     [1002, "李四", "女", "lisi@example.com", "10987654321", {"物理": "A-", "化学": "B"}]

# 打印信息
# for student in students_list:
#     print(f"学号: {student[0]}, 姓名: {student[1]}, 性别: {student[2]}, 邮箱: {student[3]}, 电话: {student[4]}, 选课信息: {student[5]}")


# 使用 dict 存储学生信息
# students_dict = {
#     1001: {
#         "姓名": "张三",
#         "性别": "男",
#         "邮箱": "zhangsan@example.com",
#         "电话": "12345678901",
#         "选课信息": {"数学": "A", "英语": "B+"}
#     },
#     1002: {
#         "姓名": "李四",
#         "性别": "女",
#         "邮箱": "lisi@example.com",
#         "电话": "10987654321",
#         "选课信息": {"物理": "A-", "化学": "B"}
#     }
# }
#
# # 打印信息
# for sid, info in students_dict.items():
#     print(f"学号: {sid}, 姓名: {info['姓名']}, 性别: {info['性别']}, 邮箱: {info['邮箱']}, 电话: {info['电话']}, 选课信息: {info['选课信息']}")
