import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x_np = np.array(x, dtype = float)
    y_np = np.array(y, dtype = float)

    res = np.sum(np.abs(x_np - y_np))
    return res