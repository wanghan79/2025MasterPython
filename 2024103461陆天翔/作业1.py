import time
import random as rnd


def benchmark_list_modification(data):
    start = time.perf_counter()
    for iteration in range(10_000):
        row = rnd.randrange(10_000)
        col = rnd.randrange(10_000)
        value = rnd.randint(0, 9)
        data[row][col] = value
    return time.perf_counter() - start


def benchmark_tuple_conversion(immutable_data):
    timer_start = time.perf_counter()
    for count in range(10_000):
        x = rnd.randrange(10_000)
        y = rnd.randrange(10_000)
        n = rnd.randint(0, 9)

        mutable = [list(inner) for inner in immutable_data]
        mutable[x][y] = n
        immutable_data = tuple(tuple(item) for item in mutable)
    return time.perf_counter() - timer_start


large_list = [[j for j in range(10000)] for i in range(10000)]
large_tuple = tuple(tuple(k for k in range(10000)) for l in range(10000))

print("Tuple processing time:", benchmark_tuple_conversion(large_tuple))
print("List processing time: ", benchmark_list_modification(large_list))