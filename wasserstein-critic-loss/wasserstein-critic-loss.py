import numpy as np

def wasserstein_critic_loss(real_scores: np.ndarray, fake_scores: np.ndarray) -> float:
    fake_mean = np.mean(fake_scores)
    
    # Compute mean of real scores
    real_mean = np.mean(real_scores)
    
    # Wasserstein critic loss
    loss = fake_mean - real_mean
    
    return loss