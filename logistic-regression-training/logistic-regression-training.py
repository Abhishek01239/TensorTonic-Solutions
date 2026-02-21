import numpy as np

# ----- Helper: numerically stable sigmoid -----
def _sigmoid(z):
    # stable sigmoid to avoid overflow
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )

# ----- Training function -----
def train_logistic_regression(X, y, lr=0.01, steps=1000):
    """
    Train binary logistic regression using gradient descent.

    Parameters:
    X : numpy array (N, D)
    y : numpy array (N,) or (N,1) with values 0 or 1
    lr : learning rate
    steps : number of gradient descent iterations

    Returns:
    (w, b)
    w -> shape (D,)
    b -> float
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    N, D = X.shape

    # initialize parameters
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        # ----- forward pass -----
        z = X @ w + b          # (N,)
        p = _sigmoid(z)        # predicted probabilities

        # ----- gradients -----
        error = p - y          # (N,)
        dw = (X.T @ error) / N
        db = np.sum(error) / N

        # ----- gradient descent update -----
        w -= lr * dw
        b -= lr * db

    return w, float(b)