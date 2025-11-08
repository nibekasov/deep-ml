import numpy as np

def discounted_return(rewards, gamma):
    """
    Compute the discounted return for a given list of rewards.
    Args:
      rewards (list of float): sequence of rewards R_{t+1}, R_{t+2}, ...
      gamma (float): discount factor (0 <= gamma <= 1)
    Returns:
      float: discounted return G_t
    """
    # Your code here
    rewards = np.asarray(rewards, dtype=float)
    timesteps = np.arange(len(rewards))
    discounts = gamma ** timesteps
    return float(np.sum(discounts * rewards))