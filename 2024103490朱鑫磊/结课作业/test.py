import time
from compute_clash_statistics import compute_clash_statistics_cpu
from compute_clash_statistics import compute_clash_statistics_gpu

start = time.time()
total_clash, relative_clashes = compute_clash_statistics_cpu("替换为蛋白质文件 .pdb")
end = time.time()

print(f"cpu total time: {end - start}")

print("")

start = time.time()
total_clash, relative_clashes = compute_clash_statistics_gpu("替换为蛋白质文件 .pdb")
end = time.time()

print(f"gpu total time: {end - start}")
