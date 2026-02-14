def maxpool_forward(X, pool_size, stride):
    H = len(X)
    W = len(X[0])
    p = pool_size
    s = stride

    H_out = (H-p)//s+1
    W_out = (W-p) // s+1

    output = [[0 for _ in range(W_out)] for _ in range(H_out)]

    for i in range(H_out):
        for j in range(W_out):
            max_val = float("-inf")

            for a in range(p):
                for b in range(p):
                    max_val = max(max_val, X[i*s +a][j*s + b])
            output[i][j] = max_val

    return output