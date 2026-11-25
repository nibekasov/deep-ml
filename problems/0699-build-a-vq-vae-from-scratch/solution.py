import numpy as np

def vq_vae_loss(
    x: np.ndarray,
    z_e: np.ndarray,
    x_recon: np.ndarray,
    codebook: np.ndarray,
    beta: float
) -> float:

    # 1. Compute pairwise squared Euclidean distances
    distances = np.sum(
        (z_e[:, None, :] - codebook[None, :, :]) ** 2,
        axis=-1
    )

    # 2. Find nearest codebook embeddings
    indices = np.argmin(distances, axis=1)
    z_q = codebook[indices]

    # 3. Compute loss components
    recon_loss = np.mean((x - x_recon) ** 2)
    codebook_loss = np.mean((z_e - z_q) ** 2)
    commitment_loss = np.mean((z_e - z_q) ** 2)

    # 4. Total loss
    return float(
        recon_loss + codebook_loss + beta * commitment_loss
    )
