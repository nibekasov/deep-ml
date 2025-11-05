import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    """
    Perform one AdamW optimizer step.
    Args:
      w: np.ndarray – parameter vector
      g: np.ndarray – gradient vector
      m: np.ndarray – first moment (mean of gradients)
      v: np.ndarray – second moment (mean of squared gradients)
      t: int – current timestep (starting from 1)
      lr: float – learning rate
      beta1, beta2: floats – decay coefficients
      epsilon: float – small constant for numerical stability
      weight_decay: float – decoupled weight decay coefficient
    Returns:
      (w_new, m_new, v_new)
    """

    # 1. Update biased first and second moment estimates
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * (g ** 2)

    # 2. Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # 3. Apply decoupled weight decay (directly on weights)
    w = w - lr * weight_decay * w

    # 4. 