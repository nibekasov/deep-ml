import numpy as np

def mae(y_true, y_pred):
	"""
	Calculate Mean Absolute Error between two arrays.

	Parameters:
	y_true (numpy.ndarray): Array of true values
    y_pred (numpy.ndarray): Array of predicted values

	Returns:
	float: Mean Absolute Error rounded to 3 decimal places
	"""
	# Преобразуем к numpy-массивам
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Проверки на корректность
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return np.nan

    # Формула: MAE = mean(|y_true - y_pred|)
    mae_val = np.mean(np.abs(y_true - y_pred))

    return round(mae_val, 3)