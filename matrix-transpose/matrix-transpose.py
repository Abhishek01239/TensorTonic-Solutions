import numpy as np

def matrix_transpose(A):
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    n,m = A.shape

    AT = np.zeros((m,n), dtype = A.dtype)

    for i in range(n):
        for j in range(m):
            AT[j, i] = A[i,j]

    return AT
    