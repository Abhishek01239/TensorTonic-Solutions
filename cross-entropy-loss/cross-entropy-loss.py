import numpy as np

def cross_entropy_loss(y_true, y_pred):
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Number of samples
    N = y_true.shape[0]

    # Select probabilities of correct classes
    correct_probs = y_pred[np.arange(N), y_true]

    # Compute loss
    loss = -np.log(correct_probs)

    # Return average loss
    return np.mean(loss)