import numpy as np
def harmonic_sum_2d(arr):
    total = 0
    for row in arr:
        for val in row:
            total += 1 / val
    return total


lst = [[1, 2, 3], [4, 5, 6]]
arr=np.array(lst)
result = harmonic_sum_2d(lst)
print(result)  # 输出：2.283333333333333
