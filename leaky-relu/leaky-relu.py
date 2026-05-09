import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    res = []
    for i in x:
        if i >= 0:
            res.append(i)
        else:
            res.append(alpha * i)
    res = np.array(res)
    return res