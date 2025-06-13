import time
import random

tuple1=tuple(tuple(range(10000)) for item in range(10000))
list1=list(list(range(10000)) for item in range(10000))

def change(data,items):
    start_time=time.time()
    for i in range(items):
        x,y=random.randint(0,items-1),random.randint(0,items-1)
        if type(data)==list:
            data[x][y]=random.randint(0,items-1)
        else:
            row = list(data[x])
            row[y] = random.randint(0,items-1)
            data = data[:x] + (tuple(row),) + data[x+1:]
    end_time=time.time()    
    return end_time-start_time

print('tupel_time=',change(tuple1,10000))
print('list_time=',change(list1,10000))

