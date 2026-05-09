import numpy as np

def expected_value_discrete(x, p):
    x_np = np.array(x, dtype=np.float64)
    p_np = np.array(p, dtype=np.float64)

    # Check shapes match
    if x_np.shape != p_np.shape:
        raise ValueError("Shapes of x and p must match")

    # Check probabilities sum to 1
    if not np.isclose(np.sum(p_np), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1")

    # Expected value
    expected = np.sum(x_np * p_np)

    return float(expected)