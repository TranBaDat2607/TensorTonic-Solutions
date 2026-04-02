import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x_np, y_np = np.array(x), np.array(y)
    return np.sqrt(np.sum((x_np - y_np) ** 2))