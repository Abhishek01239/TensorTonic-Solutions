import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    T = scores.shape[-1]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    return np.where(mask, mask_value, scores).astype(float)