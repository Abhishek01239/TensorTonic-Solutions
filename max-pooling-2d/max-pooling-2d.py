def max_pooling_2d(X, pool_size):
    if not X or not X[0]:
        return []

    H = len(X)
    W = len(X[0])
    p = pool_size

    H_out = H // p
    W_out = W // p

    result = []

    for i in range(H_out):
        row = []
        for j in range(W_out):

            max_val = float("-inf")

            for a in range(p):
                for b in range(p):
                    val = X[i*p + a][j*p + b]
                    if val > max_val:
                        max_val = val

            row.append(max_val)

        result.append(row)

    return result
