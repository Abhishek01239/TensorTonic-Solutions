import numpy as np

def conv2d(x, W, b):

    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape

    # Output size (valid convolution)
    H_out = H - KH + 1
    W_out = W_in - KW + 1

    # Create sliding windows using NumPy stride trick
    shape = (N, C_in, H_out, W_out, KH, KW)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2],
        x.strides[3],
        x.strides[2],
        x.strides[3],
    )

    patches = np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides
    )

    # patches: (N, C_in, H_out, W_out, KH, KW)
    # W:       (C_out, C_in, KH, KW)

    # Perform convolution using tensor contraction
    y = np.tensordot(
        patches, W,
        axes=([1, 4, 5], [1, 2, 3])
    )
    # Result shape: (N, H_out, W_out, C_out)

    # Rearrange to (N, C_out, H_out, W_out)
    y = np.moveaxis(y, -1, 1)

    # Add bias
    y = y + b.reshape(1, C_out, 1, 1)

    return y.astype(float)