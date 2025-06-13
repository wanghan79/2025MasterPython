import time
import random


ROWS = 10000
COLS = 10000
MODIFICATIONS = 10000


print("Creating list matrix...")
list_matrix = [[0] * COLS for _ in range(ROWS)]


print("Starting list modifications...")
start_time = time.time()
for _ in range(MODIFICATIONS):
    i = random.randint(0, ROWS - 1)
    j = random.randint(0, COLS - 1)
    list_matrix[i][j] = 1 
list_time = time.time() - start_time
print(f"List modification time: {list_time:.4f} seconds")


print("Creating tuple matrix...")
tuple_matrix = tuple(tuple(0 for _ in range(COLS)) for _ in range(ROWS))


print("Starting tuple modifications...")
start_time = time.time()

temp_matrix = list(tuple_matrix)
for _ in range(MODIFICATIONS):
    i = random.randint(0, ROWS - 1)
    j = random.randint(0, COLS - 1)

   
    row_list = list(temp_matrix[i])
    row_list[j] = 1
    new_row = tuple(row_list)


    temp_matrix[i] = new_row


tuple_matrix = tuple(temp_matrix)
tuple_time = time.time() - start_time
print(f"Tuple modification time: {tuple_time:.4f} seconds")


print("\nPerformance Comparison:")
print(f"List was {tuple_time / list_time:.1f}x faster than tuple")
print(f"List time: {list_time:.4f} sec\nTuple time: {tuple_time:.4f} sec")
