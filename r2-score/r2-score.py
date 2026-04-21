import numpy as np

def r2_score(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if np.all(y_true == y_true[0]):
        return 1.0 if np.all(y_true == y_pred) else 0.0
    y_bar = np.mean(y_true)
    r2 = 1 - (np.sum((y_true - y_pred)**2))/ (np.sum((y_true - y_bar)**2))

    return float(r2)