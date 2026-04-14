import numpy as np

def wasserstein_critic_loss(real_scores: np.ndarray, fake_scores: np.ndarray) -> float:
    fake_mean = np.mean(fake_scores)
    
    real_mean = np.mean(real_scores)
    
    loss = fake_mean - real_mean
    
    return loss