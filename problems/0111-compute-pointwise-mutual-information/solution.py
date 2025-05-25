import numpy as np
import math

def compute_pmi(count_xy, count_x, count_y, total_count):
    """
    Computes the Pointwise Mutual Information (PMI) between two events.
    
    Args:
        count_xy (int): Count of joint occurrence of x and y
        count_x (int): Count of event x
        count_y (int): Count of event y
        total_count (int): Total number of samples
    
    Returns:
        float: PMI value, or None if PMI is undefined
    """
    # Convert counts to probabilities
    p_x = count_x / total_count
    p_y = count_y / total_count
    p_xy = count_xy / total_count

    # Avoid division by zero or log of zero
    if p_xy == 0 or p_x == 0 or p_y == 0:
        return None  # PMI is undefined
    
    # Compute PMI
    pmi = math.log2(p_xy / (p_x * p_y))
    # If it's effectively an integer, return as int
    if abs(pmi - round(pmi)) < 1e-9:
        return int(round(pmi))
    return round(pmi, 3)