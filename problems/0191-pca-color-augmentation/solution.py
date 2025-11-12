import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Apply PCA color augmentation to an RGB image (AlexNet-style).

    Args:
        image: RGB image of shape (H, W, 3), dtype uint8 (values in [0, 255])
        alpha: Array of 3 random coefficients for principal components

    Returns:
        Augmented image of shape (H, W, 3), dtype float64 (not rounded to uint8)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.shape != (3,):
        raise ValueError("alpha must be a 1D array of shape (3,)")

    img = image.astype(np.float64)
    H, W, _ = img.shape
    X = img.reshape(-1, 3)

    # Mean-center
    mean_rgb = X.mean(axis=0)
    X_centered = X - mean_rgb

    # Covariance and eigen decomposition
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.