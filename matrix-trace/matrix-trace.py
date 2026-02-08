import numpy as np

def matrix_trace(A):
    A = np.asarray(A, dtype=float)

    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    return float(np.trace(A))
