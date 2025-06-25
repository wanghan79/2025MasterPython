import time
import random

array_matrix = [[0] * 10000 for _ in range(10000)]
immutable_matrix = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))

start = time.time()
for _ in range(10000):
    x = random.randint(0, 9999)
    y = random.randint(0, 9999)
    array_matrix[x][y] = random.randint(1, 100)
end = time.time()
array_time = end - start

start = time.time()
for _ in range(10000):
    x = random.randint(0, 9999)
    y = random.randint(0, 9999)
    temp_row = list(immutable_matrix[x])
    temp_row[y] = random.randint(1, 100)
    immutable_matrix = immutable_matrix[:x] + (tuple(temp_row),) + immutable_matrix[x+1:]
end = time.time()
immutable_time = end - start

print(f"Array modification time: {array_time:.2f} seconds")
print(f"Immutable modification time: {immutable_time:.2f} seconds")