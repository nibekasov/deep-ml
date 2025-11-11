import numpy as np

def rmse(y_true, y_pred):
	# Write your code here

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if y_true.shape != y_pred.shape or y_true.size == 0:
        return np.nan

    mse = np.mean((y_true - y_pred) ** 2)
    rmse_res = np.sqrt(mse)

	return round(rmse_res,3)
