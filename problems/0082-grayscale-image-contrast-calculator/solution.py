import numpy as np

def calculate_contrast(img):
    """
    Calculate the contrast of a grayscale image by taking the difference 
    between the maximum and minimum pixel values, and return it 
    rounded to 3 decimal places.

    Args:
        img (numpy.ndarray): 2D array (H x W) representing a grayscale image 
                             with pixel values between 0 and 255.

    Returns:
        float: The contrast value, rounded to 3 decimal places.
    """
    max_pixel = np.max(img)
    min_pixel = np.min(img)

    contrast = max_pixel - min_pixel
    return int(contrast)
