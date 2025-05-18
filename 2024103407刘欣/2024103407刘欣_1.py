import time
import random


list_matrix = [[0] * 10000 for _ in range(10000)]
tuple_matrix = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))

start_time = time.time()
for _ in range(10000):
    i = random.randint(0, 9999)
    j = random.randint(0, 9999)
    list_matrix[i][j] = random.randint(1, 100)
end_time = time.time()
list_time = end_time - start_time

start_time = time.time()
for _ in range(10000):
    i = random.randint(0, 9999)
    j = random.randint(0, 9999)
    temp_list = list(tuple_matrix[i])
    temp_list[j] = random.randint(1, 100)
    tuple_matrix = tuple_matrix[:i] + (tuple(temp_list),) + tuple_matrix[i+1:]
end_time = time.time()
tuple_time = end_time - start_time

print(f"List modification time: {list_time:.2f} seconds")
print(f"Tuple modification time: {tuple_time:.2f} seconds")