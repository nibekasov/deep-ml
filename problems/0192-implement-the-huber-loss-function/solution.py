import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = np.abs(y_true - y_pred)

    # Quadratic (MSE-like) region
    quadratic = 0.5 * (error ** 2)
    # Linear (MAE-like) region
    linear = delta * (error - 0.5 * delta)

    # Apply condition elementwise
    loss = np.where(error <= delta, quadratic, linear)

    return float(np.mean(loss))