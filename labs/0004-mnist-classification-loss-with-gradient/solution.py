import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction="mean", **kwargs):
    """
    preds: [N, C] — probabilities (already softmaxed)
    target: [N]   — integer class indices in [0, C-1]
    reduction: "mean" | "sum" | "none"
    Returns:
        loss: float (или вектор (N,) при reduction="none")
        d_preds: [N, C] — dL/dp, gradient w.r.t probabilities
    """

    # 1) Валидация
    assert preds.ndim == 2
    N, C = preds.shape
    assert target.shape == (N,)
    assert np.issubdtype(target.dtype, np.integer)
    assert np.all(target >= 0) and np.all(target < C)

    # 2) Численная стабильность
    eps = 1e-12
    p = np.clip(preds, eps, 1.0)  # копия/прослойка, не мутируем вход

    rows = np.arange(N)

    # 3) Пер-сэмпловая кросс-энтропия: L_i = -log p_{i, y_i}
    correct = p[rows, target]       # shape (N,)
    per_sample = -np.log(correct)   # shape (N,)

    # 4) Reduction
    if reduction == "mean":
        loss = float(per_sample.mean())
        scale = 1.0 / N
    elif reduction == "sum":
        loss = float(per_sample.sum())
        scale = 1.0
    elif reduction == "none":
        loss = per_sample
        scale = 1.0
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    # 5) Градиент по вероятностям
    # dL_i/dp_{i,c} = -1 / p_{i,y_i} если c == y_i, иначе 0
    d_preds = np.zeros_like(p)
    d_preds[rows, target] = -1.0 / correct

    # 6) Масштаб под reduction
    d_preds *= scale

    return loss, d_preds
