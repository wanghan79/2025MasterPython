# coding=utf-8
import time
import random

def run_list(mat):
    t0 = time.time()
    for _ in range(10000):
        r = random.randrange(10000)
        c = random.randrange(10000)
        v = random.randrange(10)
        mat[r][c] = v
    return time.time() - t0

def run_tuple(mat):
    t0 = time.time()
    curr = mat
    for _ in range(10000):
        r = random.randrange(10000)
        c = random.randrange(10000)
        v = random.randrange(10)
        tmp = [list(row) for row in curr]
        tmp[r][c] = v
        curr = tuple(tuple(row) for row in tmp)
    return time.time() - t0

if __name__ == "__main__":
    N = 10000
    L = [[i for i in range(N)] for _ in range(N)]
    T = tuple(tuple(i for i in range(N)) for _ in range(N))

    print(run_tuple(T))
    print(run_list(L))
