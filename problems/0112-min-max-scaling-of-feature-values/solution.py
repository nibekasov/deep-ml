def min_max(x: list[int]) -> list[float]:
    if not x:
        return []
    
    xmin, xmax = min(x), max(x)
    if xmin == xmax:
        return [0.0 for _ in x]  # все элементы одинаковы

    return [round((xi - xmin) / (xmax - xmin), 4) for xi in x]
