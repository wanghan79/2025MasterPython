import time
import random

# 生成 100x100 的列表
list1 = [[0 for _ in range(100)] for _ in range(100)]

start = time.time()
for _ in range(10000):
    i = random.randint(0, 99)
    j = random.randint(0, 99)
    list1[i][j] = 1  # 直接修改元素
end = time.time()
print(f"修改列表耗时: {end - start:.4f} 秒")

# 生成 100x100 的元组
tuple1 = tuple(tuple(0 for _ in range(100)) for _ in range(100))

start = time.time()
for _ in range(10000):
    i = random.randint(0, 99)
    j = random.randint(0, 99)
    # 每次重建整个元组
    new_row = tuple(1 if (x == j) else val for x, val in enumerate(tuple1[i]))
    tuple1 = tuple(new_row if (y == i) else row for y, row in enumerate(tuple1))
end = time.time()
print(f"修改元组耗时: {end - start:.4f} 秒")
