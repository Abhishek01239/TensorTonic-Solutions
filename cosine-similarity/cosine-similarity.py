import numpy as np

def cosine_similarity(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")

    if x.shape[0] != y.shape[0]:
        raise ValueError("Vectors must have the same length")

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    # Handle zero-vector case
    if norm_x == 0 or norm_y == 0:
        return 0.0

    return float(np.dot(x, y) / (norm_x * norm_y))
