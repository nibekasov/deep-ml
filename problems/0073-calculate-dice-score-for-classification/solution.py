
import numpy as np

def dice_score(y_true, y_pred):
    # Convert inputs to NumPy arrays (in case they aren't already)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate True Positives (intersection of 1s)
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    
    # Sum of positives in each array
    sum_true = np.sum(y_true == 1)
    sum_pred = np.sum(y_pred == 1)
    
    # Handle edge case where both sums are zero (no positives)
    if sum_true + sum_pred == 0:
        return 0.0
    
    # Dice formula
    dice = (2 * intersection) / (sum_true + sum_pred)
    
    return round(dice, 3)
