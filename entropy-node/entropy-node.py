import numpy as np

def entropy_node(y):
    # Convert to numpy array
    y = np.asarray(y)
    
    # Handle empty node
    if y.size == 0:
        return 0.0
    
    # Count occurrences of each class
    _, counts = np.unique(y, return_counts=True)
    
    # Compute probabilities
    probabilities = counts / counts.sum()
    
    # Compute entropy (ignore zero probabilities for stability)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return float(entropy)