import math

def log_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    y_clip = np.clip(y_pred, eps, 1- eps)

    losses = -(y_true*np.log(y_clip) + (1-y_true) * np.log(1-y_clip))

    return losses.tolist()