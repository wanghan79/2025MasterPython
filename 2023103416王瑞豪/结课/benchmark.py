import time
import random
import sys

class DataBenchmark:
    def __init__(self, size: int = 1000, modify_rounds: int = 10000):
        self.size = size
        self.modify_rounds = modify_rounds

    def build_list_matrix(self):
        return [[0 for _ in range(self.size)] for _ in range(self.size)]

    def build_tuple_matrix(self):
        return tuple(tuple(0 for _ in range(self.size)) for _ in range(self.size))

    def run_list_test(self):
        print(f"\n[LIST TEST] size={self.size}×{self.size}, modify={self.modify_rounds}")
        start_time = time.time()
        matrix = self.build_list_matrix()
        build_time = time.time()

        for _ in range(self.modify_rounds):
            i = random.randint(0, self.size - 1)
            j = random.randint(0, self.size - 1)
            matrix[i][j] += 1

        end_time = time.time()
        return {
            "structure": "list",
            "build_time": build_time - start_time,
            "modify_time": end_time - build_time,
            "total_time": end_time - start_time,
            "memory_MB": sys.getsizeof(matrix) / 1024 / 1024
        }

    def run_tuple_test(self):
        print(f"\n[TUPLE TEST] size={self.size}×{self.size}, modify={self.modify_rounds}")
        start_time = time.time()
        matrix = self.build_tuple_matrix()
        build_time = time.time()

        # NOTE: tuple 是不可变的，所以不能原地修改，只能模拟修改时间
        for _ in range(self.modify_rounds):
            i = random.randint(0, self.size - 1)
            j = random.randint(0, self.size - 1)
            _ = matrix[i][j] + 1  # 模拟读取加修改，但不赋值

        end_time = time.time()
        return {
            "structure": "tuple",
            "build_time": build_time - start_time,
            "modify_time": end_time - build_time,
            "total_time": end_time - start_time,
            "memory_MB": sys.getsizeof(matrix) / 1024 / 1024
        }

    def compare(self):
        list_result = self.run_list_test()
        tuple_result = self.run_tuple_test()

        print("\n--- Benchmark Result (seconds) ---")
        print("{:<10} | {:>10} | {:>12} | {:>10} | {:>10}".format(
            "Type", "Build", "Modify", "Total", "Memory(MB)"
        ))
        for res in [list_result, tuple_result]:
            print("{:<10} | {:>10.4f} | {:>12.4f} | {:>10.4f} | {:>10.2f}".format(
                res["structure"], res["build_time"], res["modify_time"], res["total_time"], res["memory_MB"]
            ))


if __name__ == "__main__":
    bench = DataBenchmark(size=1000, modify_rounds=10000)
    bench.compare()
