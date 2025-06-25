import timeit
import random
def create_large_list_matrix(n):
    return [[random.randint(0, 100) for _ in range(n)] for _ in range(n)]
def create_large_tuple_matrix(n):
    return tuple(tuple(random.randint(0, 100) for _ in range(n)) for _ in range(n))

def test_list_modification(matrix, iterations):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    for _ in range(iterations):
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        matrix[r][c] = random.randint(0, 100)
    return matrix

def test_tuple_modification(matrix, iterations):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    for _ in range(iterations):
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        row = list(matrix[r])
        row[c] = random.randint(0, 100)
        matrix = list(matrix)
        matrix[r] = tuple(row)
        matrix = tuple(matrix)
    return matrix


def run_tests():
    n = 5000
    iterations = 5000

    list_matrix = create_large_list_matrix(n)

    tuple_matrix = create_large_tuple_matrix(n)

    start_time = timeit.default_timer()
    test_list_modification(list_matrix, iterations)
    list_time = timeit.default_timer() - start_time


    start_time = timeit.default_timer()
    test_tuple_modification(tuple_matrix, iterations)
    tuple_time = timeit.default_timer() - start_time

    print(f"列表修改时间: {list_time:.2f} 秒")
    print(f"元组修改时间: {tuple_time:.2f} 秒")


run_tests()