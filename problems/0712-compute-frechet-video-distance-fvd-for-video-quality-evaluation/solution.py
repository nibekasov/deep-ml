import numpy as np

def compute_fvd(features_real, features_gen) -> float:
    real = np.asarray(features_real, dtype=float)
    gen = np.asarray(features_gen, dtype=float)

    # Средние
    mu_r = real.mean(axis=0)
    mu_g = gen.mean(axis=0)

    # Ковариации
    cov_r = np.atleast_2d(np.cov(real, rowvar=False))
    cov_g = np.atleast_2d(np.cov(gen, rowvar=False))

    # Расстояние между средними
    mean_dist = np.sum((mu_r - mu_g) ** 2)

    # Собственные значения произведения ковариаций
    eigenvalues = np.linalg.eigvals(cov_r @ cov_g)
    eigenvalues = np.clip(eigenvalues.real, 0, None)

    # Расстояние между ковариациями
    cov_dist = (
        np.trace(cov_r)
        + np.trace(cov_g)
        - 2 * np.sqrt(eigenvalues).sum()
    )

    return float(max(mean_dist + cov_dist, 0.0))