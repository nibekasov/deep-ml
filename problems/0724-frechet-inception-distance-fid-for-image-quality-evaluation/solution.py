import numpy as np

def fid_score(real_features, generated_features) -> float:
    real = np.asarray(real_features, dtype=float)
    generated = np.asarray(generated_features, dtype=float)

    mu_r = real.mean(axis=0)
    mu_g = generated.mean(axis=0)

    centered_r = real - mu_r
    centered_g = generated - mu_g

    cov_r = centered_r.T @ centered_r / (real.shape[0] - 1)
    cov_g = centered_g.T @ centered_g / (generated.shape[0] - 1)

    eigvals, eigvecs = np.linalg.eigh(cov_r)
    sqrt_r = (
        eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))
    ) @ eigvecs.T

    middle = sqrt_r @ cov_g @ sqrt_r
    middle = (middle + middle.T) / 2
    middle_eigvals = np.linalg.eigvalsh(middle)

    trace_sqrt = np.sqrt(np.clip(middle_eigvals, 0.0, None)).sum()

    mean_diff = mu_r - mu_g
    fid = (
        mean_diff @ mean_diff
        + np.trace(cov_r)
        + np.trace(cov_g)
        - 2 * trace_sqrt
    )

    return float(max(fid, 0.0))