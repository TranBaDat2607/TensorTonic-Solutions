import numpy as np

def normalize_3d(v):
    v = np.array(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v 
        return v / norm
        
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1
    
    return v / norm