
import time
import random

MATRIX_DIM = 10000
NUM_OPERATIONS = 10000

# --- 列表（可变）操作 ---
mutable_grid = [[0] * MATRIX_DIM for _ in range(MATRIX_DIM)]

start_time_mutable = time.perf_counter()
for _ in range(NUM_OPERATIONS):
    row_idx = random.randint(0, MATRIX_DIM - 1)
    col_idx = random.randint(0, MATRIX_DIM - 1)
    mutable_grid[row_idx][col_idx] = random.randint(1, 100)
end_time_mutable = time.perf_counter()
time_mutable_ops = end_time_mutable - start_time_mutable

immutable_grid = tuple(tuple(0 for _ in range(MATRIX_DIM)) for _ in range(MATRIX_DIM))
start_time_immutable = time.perf_counter()
for _ in range(NUM_OPERATIONS):
    row_idx = random.randint(0, MATRIX_DIM - 1)
    col_idx = random.randint(0, MATRIX_DIM - 1)
    new_value = random.randint(1, 100)

    temp_row_list = list(immutable_grid[row_idx])
    temp_row_list[col_idx] = new_value

    immutable_grid = immutable_grid[:row_idx] + \
                     (tuple(temp_row_list),) + \
                     immutable_grid[row_idx + 1:]
end_time_immutable = time.perf_counter()
time_immutable_ops = end_time_immutable - start_time_immutable
print(f"List modification time: {time_mutable_ops:.6f} seconds")
print(f"Tuple 'modification' time: {time_immutable_ops:.6f} seconds")
