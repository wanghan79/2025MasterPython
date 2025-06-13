# coding=utf-8

import time
import random

def list_test(data_list):
    start_time = time.time()
    for _ in range(10000):
        row_index = random.randint(0,9999)
        col_index = random.randint(0,9999)
        new_value = random.randint(0,9)
        data_list[row_index][col_index] = new_value
    return time.time() - start_time


def tuple_test(data_tuple):
    start_time = time.time()
    for _ in range(10000):
        row_index = random.randint(0, 9999)
        col_index = random.randint(0, 9999)
        new_value = random.randint(0, 9)

        mutable_list = list(map(list,data_tuple))
        mutable_list[row_index][col_index] = new_value
        data_tuple = tuple(map(tuple,mutable_list))
    return time.time() - start_time

list_matrix = [[i for i in range(10000)] for _ in range(10000)]

tuple_matrix = tuple(tuple(i for i in range(10000)) for _ in range(10000))

print(tuple_test(tuple_matrix))

print(list_test(list_matrix))