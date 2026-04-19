import numpy as np

def sample_var_std(x):
    x = np.array(x, dtype=float)
    
    n = x.size
    
    if n < 2:
        raise ValueError("At least 2 data points required")
    
    mean = np.mean(x)
    
    var = np.sum((x - mean) ** 2) / (n - 1)
    
    std = np.sqrt(var)
    
    return float(var), float(std)