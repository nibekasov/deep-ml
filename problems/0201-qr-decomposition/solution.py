import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using the (classical) Gram–Schmidt process.
    
    Args:
        A: m x n matrix as list of lists (rows)
    
    Returns:
        (Q, R):
          Q — m x n, столбцы ортонормированы (Q^T Q ≈ I_n)
          R — n x n, верхнетреугольная
        Обе матрицы возвращаются как list[list[float]].
    """
    # Преобразуем в numpy-массив
    A = np.array(A, dtype=float)   # shape (m, n)
    m, n = A.shape
    
    # Инициализируем Q и R
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    
    # Классический Грам–Шмидт по столбцам
    for j in range(n):
        # Берём j-й столбец A как исходный вектор
        v = A[:, j].copy()
        
        # Вычитаем проекции на уже построенные q_i (i < j)
        for i in range(j):
            # коэффициент проекции: r_ij = q_i^T a_j
            R