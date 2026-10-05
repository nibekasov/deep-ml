import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    if n_steps <= 0:
        return []

    w = np.array(w0, dtype=float).copy()
    losses = []
    n = X.shape[0]

    for k in range(n_steps):

        if n_steps == 1:
            lr = 10 ** a
        else:
            lr = 10 ** (a + (b - a) * k / (n_steps - 1))

        y_pred = X @ w

        grad = (2 / n) * X.T @ (y_pred - y)

        w = w - lr * grad

        loss = np.mean((X @ w - y) ** 2)

        losses.append(float(loss))

    return losses 
