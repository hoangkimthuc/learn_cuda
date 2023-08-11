import numpy as np
from time import time

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[20_000_000, 5])
data = arr.tolist()
print(len(data))
start = time()
def howmany_within_range(row, minimum=4, maximum=8):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
results = pool.map(howmany_within_range, [row for row in data])


# Step 3: Don't forget to close
pool.close()    

print("total time taken: ", time() - start)
print(results[:10])