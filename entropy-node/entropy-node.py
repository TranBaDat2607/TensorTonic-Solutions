import numpy as np

def entropy_node(y):
    """
    Compute entropy for a node from class labels.
    """

    y_np = np.array(y)
    _, counts = np.unique(y_np, return_counts=True)

    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)