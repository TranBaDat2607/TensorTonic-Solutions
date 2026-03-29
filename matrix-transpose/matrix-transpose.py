import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    matrix_np = np.array(A)
    return np.transpose(matrix_np)
