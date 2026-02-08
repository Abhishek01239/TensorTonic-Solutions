import numpy as np

def euclidean_distance(x, y):
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input must be 1D vector")
    
    if x.shape[0] != y.shape[0]:
        raise ValueError("Vectors have the same length")

    return float(np.linalg.norm(x-y))