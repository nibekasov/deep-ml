import numpy as np

def discounted_return(rewards, gamma):
    """
    Compute the total discounted return for a sequence of rewards.

    Args:
        rewards (list or np.ndarray): Rewards [r_0, r_1, ..., r_T-1]
        gamma (float): Discount factor (0 < gamma <= 1)

    Returns:
        float: Total discounted return
    """
    rewards = np.asarray(rewards, dtype=float)
    timesteps = np.arange(len(rewards))
    discounts = gamma ** timesteps
    return float(np.sum(discounts * rewards))
