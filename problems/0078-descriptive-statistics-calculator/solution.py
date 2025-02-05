import numpy as np

def descriptive_statistics(data):
    """
    Compute descriptive statistics using only NumPy.

    Parameters:
        data (list or np.array): List or NumPy array of numerical values.

    Returns:
        dict: Dictionary containing mean, median, mode, variance, standard deviation,
              percentiles (25th, 50th, 75th), and interquartile range (IQR).
    """
    data = np.array(data)

    # Compute mode manually using numpy (since scipy is not allowed)
    unique, counts = np.unique(data, return_counts=True)
    mode_value = unique[np.argmax(counts)]  # Most frequent value

    stats_dict = {
        "mean": round(np.mean(data), 4),
        "median": round(np.median(data), 4),
        "mode": round(mode_value, 4),
        "variance": round(np.var(data, ddof=0), 4),  # Population variance
        "standard_deviation": round(np.std(data, ddof=0), 4),  # Population standard deviation
        "25th_percentile": round(np.percentile(data