# coding=utf-8
# work1

import time
import random


def mutable_test(data_grid):
    start = time.perf_counter()
    for _ in range(10000):
        row = random.randrange(10000)
        col = random.randrange(10000)
        value = random.randrange(10)
        data_grid[row][col] = value
    return time.perf_counter() - start


def immutable_test(data_grid):
    start = time.perf_counter()
    for _ in range(10000):
        row = random.randrange(10000)
        col = random.randrange(10000)
        value = random.randrange(10)

        temp = [list(inner) for inner in data_grid]
        temp[row][col] = value
        data_grid = tuple(tuple(inner) for inner in temp)
    return time.perf_counter() - start


mutable_grid = [[j for j in range(10000)] for _ in range(10000)]
immutable_grid = tuple(tuple(j for j in range(10000)) for _ in range(10000))

immutable_time = immutable_test(immutable_grid)
mutable_time = mutable_test(mutable_grid)

print(f'不可变数据修改耗时: {immutable_time:.4f}秒')
print(f'可变数据修改耗时: {mutable_time:.4f}秒')
