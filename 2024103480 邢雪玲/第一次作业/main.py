import time
import random

def test_list_performance():
    start = time.time()
    data = [[0 for _ in range(10000)] for _ in range(10000)]
    init_time = time.time() - start

    start_mod = time.time()
    for _ in range(10000):
        i = random.randint(0, 9999)
        j = random.randint(0, 9999)
        data[i][j] = random.randint(1, 100)
    mod_time = time.time() - start_mod

    return init_time, mod_time


def test_tuple_performance():
    start = time.time()
    data = tuple(tuple(0 for _ in range(10000)) for _ in range(10000))
    init_time = time.time() - start

    start_mod = time.time()
    current_data = data
    for _ in range(10000):
        i = random.randint(0, 9999)
        j = random.randint(0, 9999)
        new_value = random.randint(1, 100)
        new_row = current_data[i][:j] + (new_value,) + current_data[i][j + 1:]
        current_data = current_data[:i] + (new_row,) + current_data[i + 1:]
    mod_time = time.time() - start_mod

    return init_time, mod_time



print("Testing list performance...")
list_init, list_mod = test_list_performance()
print(f"List init: {list_init:.2f}s | 10k mods: {list_mod:.2f}s")

print("\nTesting tuple performance...")
tuple_init, tuple_mod = test_tuple_performance()
print(f"Tuple init: {tuple_init:.2f}s | 10k mods: {tuple_mod:.2f}s")
