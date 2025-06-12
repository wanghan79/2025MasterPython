# list tuple 比较

import time

size = 10000

#Tuple
tuple1 = tuple(tuple(0 for _ in range(size)) for _ in range(size))

def update_element(element):
    return element + 1

start_time = time.time()
tuple1 = tuple(tuple(update_element(element) for element in row) for row in tuple1)
end_time = time.time()
print("Tuple:")
print("used time: " + str(end_time - start_time))

#List
list1 = list(list(0 for _ in range(size)) for _ in range(size))

start_time = time.time()
for row in list1:
    for element in row:
        element = 1
end_time = time.time()
print("List:")
print("used time: " + str(end_time - start_time))
"""
size = 10000 * 10000
Tuple:
used time: 6.807703495025635
List:
used time: 3.2258243560791016
"""