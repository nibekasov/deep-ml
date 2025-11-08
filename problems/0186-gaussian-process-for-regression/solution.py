# -*- coding: utf-8 -*-
import numpy as np

# -------------------------- Kernels ---------------------------------

def _pairwise_sq_dists(X1, X2):
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    X1_sq = np.sum(X1**2, axis=1, keepdims=True)       # (n1,1)
    X2_sq = np.sum(X2**2, axis=1, keepdims=True).T     # (1,n2)
    return np.maximum(X1_sq + X2_sq - 2.0 * X1 @ X2.T, 0.0)

def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5, sigma=1.0):
    r = np.linalg.norm(x - x_prime) / (length_scale + 1e-15)
    if nu == 0.5:
        return (sigma**2) * np.exp(-r)
    elif nu == 1.5:
        c = np.sqrt(3.0) * r
        return (sigma**2) * (1.0 + c) * np.exp(-c)
    elif nu == 2.5:
        c = np.sqrt(5.0) * r
        return (sigma**2) * (1.0 + c + (5.0/3.0) * r**2) * np.exp(-c)
    else:
        raise NotImplementedError("Matérn implemented only for ν in {0.5, 1.5, 2.5}.")

def rbf_kernel(x: np.ndarray, x_prime, sigma=1.0, length_scale=1.0):