import numpy as np

def tanh(x):
    x = np.array(x, dtype=float)
    
    result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    return result