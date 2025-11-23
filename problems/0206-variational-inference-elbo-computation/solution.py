import numpy as np
import math

def compute_elbo(x: list[float], q_mean: float, q_std: float, 
                 prior_mean: float, prior_std: float,
                 likelihood_std: float, n_samples: int = 1000) -> float:
    """
    Compute the Evidence Lower Bound (ELBO) for variational inference.
    
    Args:
        x: Observed data points
        q_mean: Mean of variational distribution q(z)
        q_std: Standard deviation of variational distribution q(z)
        prior_mean: Mean of prior p(z)
        prior_std: Standard deviation of prior p(z)
        likelihood_std: Standard deviation of likelihood p(x|z)
        n_samples: Number of Monte Carlo samples from q(z)
    
    Returns:
        ELBO value
    """
    x_arr = np.array(x, dtype=float)   # shape: (n_x,)

    # Фиксируем сид для воспроизводимости (важно для точного совпадения с тестами)
    np.random.seed(8)

    # 1) Сэмплируем z ~ q(z) = N(q_mean, q_std^2)
    z = np.random.normal(loc=q_me