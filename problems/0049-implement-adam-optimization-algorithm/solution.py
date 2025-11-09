import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    """
    Adam optimizer.

    Args:
        f (callable): целевая функция f(x) (не обязательно вызывать внутри оптимизатора).
        grad (callable): градиентная функция grad(x) -> np.ndarray той же формы, что x.
        x0 (array-like): начальные параметры.
        learning_rate (float): шаг обучения (alpha).
        beta1 (float): коэффициент эксп. усреднения 1-го момента.
        beta2 (float): коэффициент эксп. усреднения 2-го момента.
        epsilon (float): числ. стабилизация.
        num_iterations (int): число итераций.

    Returns:
        np.ndarray: оптимизированные параметры.
    """
	x = np.asarray(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, int(num_iterations) + 1):
        g = np.asarray(grad(x), dtype=float)

        m = beta1 * m + (1.0 - beta1) * g
        v = b