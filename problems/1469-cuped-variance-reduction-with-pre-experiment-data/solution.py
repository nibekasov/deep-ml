import numpy as np


def cuped_theta(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    xc = x - x.mean()
    yc = y - y.mean()

    denominator = np.sum(xc ** 2)

    if denominator == 0:
        return 0.0

    return float(np.sum(xc * yc) / denominator)


def cuped_adjust(y, x, theta, x_mean):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    return y - theta * (x - x_mean)


def ab_estimate(y_t, y_c):
    y_t = np.asarray(y_t, dtype=float)
    y_c = np.asarray(y_c, dtype=float)

    diff = y_t.mean() - y_c.mean()

    se = np.sqrt(
        y_t.var(ddof=1) / y_t.size
        + y_c.var(ddof=1) / y_c.size
    )

    return float(diff), float(se)


def cuped_ab(y_t, x_t, y_c, x_c):
    y = np.concatenate([y_t, y_c]).astype(float)
    x = np.concatenate([x_t, x_c]).astype(float)

    theta = cuped_theta(y, x)
    x_mean = x.mean()

    adj_t = cuped_adjust(y_t, x_t, theta, x_mean)
    adj_c = cuped_adjust(y_c, x_c, theta, x_mean)

    diff_raw, se_raw = ab_estimate(y_t, y_c)
    diff_cuped, se_cuped = ab_estimate(adj_t, adj_c)

    raw_var = y.var(ddof=1)
    adj_var = np.concatenate([adj_t, adj_c]).var(ddof=1)

    variance_reduction = (
        1 - adj_var / raw_var
        if raw_var > 0
        else 0.0
    )

    result = {
        "diff_raw": diff_raw,
        "se_raw": se_raw,
        "diff_cuped": diff_cuped,
        "se_cuped": se_cuped,
        "theta": theta,
        "variance_reduction": variance_reduction,
    }

    return {
        key: round(float(value), 6)
        for key, value in result.items()
    }


def sample_size_multiplier(rho):
    return float(1 - rho ** 2)