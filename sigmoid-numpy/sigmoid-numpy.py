import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x_arr = np.array(x)
    res = 1/(1+np.exp(-x_arr))
    return res