import numpy as np

def matrix_inverse(A):
    A = np.asarray(A, dtype=float)

    if A.ndim != 2:
        return None

    n, m = A.shape

    if n != m:
        return None

    if np.isclose(np.linalg.det(A), 0.0):
        return None

    A_inv = np.linalg.inv(A)

    return A_inv
