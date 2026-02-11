import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    if seq_len < 1 or d_model < 1:
        return None

    # Positions (seq_len, 1)
    positions = np.arange(seq_len, dtype=float)[:, np.newaxis]

    # Even dimension indices (0,2,4,...)
    dims = np.arange(0, d_model, 2, dtype=float)

    # Compute angle rates: base^(2i/d_model)
    angle_rates = 1.0 / (base ** (dims / d_model))

    # Compute angle matrix
    angles = positions * angle_rates  # shape (seq_len, ceil(d_model/2))

    # Initialize PE matrix
    PE = np.zeros((seq_len, d_model), dtype=float)

    # Fill even indices with sin
    PE[:, 0::2] = np.sin(angles)

    # Fill odd indices with cos
    # Need to handle odd d_model case carefully
    num_cos_cols = PE[:, 1::2].shape[1]
    PE[:, 1::2] = np.cos(angles[:, :num_cos_cols])

    return PE
