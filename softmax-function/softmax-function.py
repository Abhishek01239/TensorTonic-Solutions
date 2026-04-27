import numpy as np

def softmax(x):
    x = np.array(x, dtype=float)

    if x.ndim == 1:
        x = x - np.max(x)                
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    elif x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    else:
        raise ValueError("Input must be 1D or 2D array")