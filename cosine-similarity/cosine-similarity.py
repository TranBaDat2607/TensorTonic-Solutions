import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a_np = np.array(a, dtype=float)
    b_np = np.array(b, dtype=float)

    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cs = np.dot(a_np, b_np) / (norm_a * norm_b)

    return float(cs)