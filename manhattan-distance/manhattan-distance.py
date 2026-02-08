import numpy as np

def manhattan_distance(x, y):
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input must be 1d Vectors")
    
    if x.shape[0] != y.shape[0]:
        raise ValueError("Vectors shoulb be have same length")
    
    return float(np.sum(np.abs(x-y)))