"""
    这个是适应度函
"""
import numpy as np


def chung_reynolds(x):
    n = np.shape(x)[1]
    f = 0
    for i in range(n):
        f = f + np.power(np.power(x[0, i], 2), 2)
    return f

