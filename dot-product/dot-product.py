import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    x_np = np.array(x)
    y_np = np.array(y)

    return np.dot(x_np, y_np)