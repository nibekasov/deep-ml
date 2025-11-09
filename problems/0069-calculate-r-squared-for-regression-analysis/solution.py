
import numpy as np

def r_squared(y_true, y_pred):
    """
    Compute the R-squared (coefficient of determination) for regression.

    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted values
    Returns:
        float: R-squared value rounded to 3 decimal places
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    ss_res = np.sum((y_true - y_pred) ** 2)           # Sum of Squared Residuals (SSR)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total Sum of Squares (SST)
    
    # Handle edge case when all y_true are identical (no variance)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return float(np.round(r2, 3))