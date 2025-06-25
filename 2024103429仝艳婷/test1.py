# coding=utf-8

import time
import random

def list_test(list):
    start_time = time.time()
    for _ in range(10000):
        index_row = random.randint(0,9999)
        index_column = random.randint(0,9999)
        num = random.randint(0,9)
        list[index_row][index_column] = num
    return time.time() - start_time


def tuple_test(tuple_):
    start_time = time.time()
    for _ in range(10000):
        index_row = random.randint(0, 9999)
        index_column = random.randint(0, 9999)
        num = random.randint(0, 9)

        list_ = list(map(list,tuple_))
        list_[index_row][index_column] = num
        tuple_ = tuple(map(tuple,tuple_))
    return time.time() - start_time

list_matrix = [[i for i in range(10000)] for _ in range(10000)]

tuple_matrix = tuple(tuple(i for i in range(10000)) for _ in range(10000))

print(tuple_test(tuple_matrix))

print(list_test(list_matrix))

