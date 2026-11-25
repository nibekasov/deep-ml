import numpy as np

def vq_ste_backward(z_e: np.ndarray, codebook: np.ndarray,
                    dL_dzq: np.ndarray, beta: float) -> np.ndarray:
    diff = z_e[:, None, :] - codebook            # (N, K, D)
    distances_sq = np.sum(diff ** 2, axis=-1)    # (N, K)
    nearest = codebook[np.argmin(distances_sq, axis=1)]  # (N, D)

    return dL_dzq + 2 * beta * (z_e - nearest)
