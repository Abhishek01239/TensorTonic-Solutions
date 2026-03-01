import numpy as np

def entropy_node(y):
    y = np.asarray(y)

    if y.size == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    
    # Compute probabilities
    probabilities = counts / counts.sum()
    
    # Stable log computation (ignore zero probabilities)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return float(entropy)