# 作业1：可变与不可变容器性能对比实验
import time
import random

# 容器维度和操作次数
DIM = 1000  # 适当减小，防止内存溢出
MODIFY_TIMES = 10000

def build_list_grid(size):
    """生成二维列表网格"""
    return [[0]*size for _ in range(size)]

def build_tuple_grid(size):
    """生成二维元组网格"""
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))

def update_list_grid(grid, times):
    """多次随机修改列表网格的元素"""
    n = len(grid)
    t0 = time.perf_counter()
    for _ in range(times):
        x = random.randrange(n)
        y = random.randrange(n)
        grid[x][y] = random.randint(1, 100)
    t1 = time.perf_counter()
    return t1 - t0

def update_tuple_grid(grid, times):
    """多次随机修改元组网格的元素（需新建元组）"""
    n = len(grid)
    t0 = time.perf_counter()
    for _ in range(times):
        x = random.randrange(n)
        y = random.randrange(n)
        row = list(grid[x])
        row[y] = random.randint(1, 100)
        grid = grid[:x] + (tuple(row),) + grid[x+1:]
    t1 = time.perf_counter()
    return t1 - t0

def run():
    print(f"创建 {DIM}x{DIM} 的二维容器...")
    l_grid = build_list_grid(DIM)
    t_grid = build_tuple_grid(DIM)
    print("\n开始计时...")
    print(f"每种容器各执行 {MODIFY_TIMES} 次随机修改\n")
    print("列表容器计时...")
    t_list = update_list_grid(l_grid, MODIFY_TIMES)
    print(f"列表耗时: {t_list:.6f} 秒")
    print("\n元组容器计时...")
    t_tuple = update_tuple_grid(t_grid, MODIFY_TIMES)
    print(f"元组耗时: {t_tuple:.6f} 秒")
    print("\n对比结果:")
    print(f"元组比列表慢 {t_tuple/t_list:.2f} 倍")
    print("\n分析: 列表可变，直接修改，效率高。元组不可变，每次都要新建，效率低。")

if __name__ == "__main__":
    run()
