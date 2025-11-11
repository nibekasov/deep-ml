import numpy as np

def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables.

    Args:
        x (list[int]): A list of binary values (0 or 1).
        y (list[int]): A list of binary values (0 or 1).

    Returns:
        float: The Phi coefficient rounded to 4 decimal places.
    """
    # Преобразуем к numpy массивам
    x = np.array(x)
    y = np.array(y)

    # Проверка длины
    if len(x) != len(y) or len(x) == 0:
        return np.nan

    # Подсчёт частот (элементов таблицы сопряжённости)
    x00 = np.sum((x == 0) & (y == 0))
    x01 = np.sum((x == 0) & (y == 1))
    x10 = np.sum((x == 1) & (y == 0))
    x11 = np.sum((x == 1) & (y == 1))

    # Числитель и знаменатель формулы
    numerator = x00 * x11 - x01 * x10
    denominator = np.sqrt(
        (x00 + x01) * (x10 + x11) * (x00 + x10) * (x01 + x11)
    )

    # Проверка деления на ноль
    if denominator == 0:
        return 