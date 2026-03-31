import numpy as np

def dropout(x, p, rng=None):

    x = np.asarray(x)

    if p == 0.0:
        dropout_pattern = np.ones_like(x, dtype=float)
        return x.astype(float), dropout_pattern

    if rng is not None:
        random_vals = rng.random(x.shape)
    else:
        random_vals = np.random.random(x.shape)

    keep_mask = random_vals >= p

    scale = 1.0 / (1.0 - p)

    dropout_pattern = keep_mask.astype(float) * scale

    output = x * dropout_pattern

    return output, dropout_pattern