import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    """
    Solve Ax = b via Gauss-Seidel iterations.

    Args:
        A (np.ndarray): квадратная матрица (m x m)
        b (np.ndarray): вектор правой части (m,)
        n (int): число итераций
        x_ini (np.ndarray | None): начальное приближение (m,). Если None -> нули.

    Returns:
        np.ndarray: приближённое решение x после n итераций
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    m, k = A.shape
    if m != k:
        raise ValueError("A must be square.")
    if b.shape[0] != m:
        raise ValueError("b must have length equal to A.shape[0].")

    x = np.zeros(m, dtype=float) if x_ini is None else np.asarray(x_ini, dtype=float).copy()

    for _ in range(int(n)):
        for i in range(m):
            aii = A[i, i]
            if aii == 0.0:
                raise ZeroDivisionError(f"Zero diagonal element at row {i}.")
            # Используем 