def rmsprop_update(params: list[float], grads: list[float], cache: list[float], 
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
    """
    Perform one RMSProp optimization update step.

    Args:
        params: List of parameter values
        grads: List of gradients for each parameter
        cache: List of cache values (moving average of squared gradients)
        lr: Learning rate
        beta: Decay rate for moving average (0..1)
        epsilon: Small constant for numerical stability

    Returns:
        (updated_params, updated_cache)
    """
    assert len(params) == len(grads) == len(cache)

    new_params = []
    new_cache = []

    for p, g, v in zip(params, grads, cache):
        # 1) обновляем cache: v_new = beta * v + (1 - beta) * g^2
        v_new = beta * v + (1.0 - beta) * (g * g)

        # 2) шаг RMSProp: p_new = p - lr * g / (sqrt(v_new) + eps)
        denom = (v_new **