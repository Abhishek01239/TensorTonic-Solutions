import numpy as np

def percentiles(x, q):
    x = np.array(x)
    q = np.array(q)

    return np.percentile(x, q, method='linear')