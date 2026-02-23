import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    # If seqs is empty → return (0,0) array
    if not seqs:
        return np.empty((0, 0), dtype=int)
    
    # Determine max length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    N = len(seqs)
    
    # Create output array filled with pad_value
    result = np.full((N, max_len), pad_value, dtype=int)
    
    # Fill values
    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]  # truncate if longer
        result[i, :len(trunc)] = trunc
    
    return result