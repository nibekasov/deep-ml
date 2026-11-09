import numpy as np

def first_frame_anchor_inject(
    history_frames: np.ndarray,
    first_frame: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    rng: np.random.Generator
) -> np.ndarray:
    sigma = rng.uniform(sigma_min, sigma_max)
    noise = rng.normal(0.0, sigma, size=first_frame.shape)
    noisy_anchor = first_frame + noise

    return np.concatenate(
        [noisy_anchor[np.newaxis, ...], history_frames],
        axis=0
    )