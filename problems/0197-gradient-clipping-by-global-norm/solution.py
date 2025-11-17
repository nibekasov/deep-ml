import math

def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
    # 1. Compute global L2 norm
    global_sum = 0.0
    for g in gradients:
        for x in g:
            global_sum += x * x
    global_norm = math.sqrt(global_sum)

    # 2. If already below threshold → return unchanged
    if global_norm <= max_norm or global_norm == 0:
        return gradients

    # 3. Compute scaling factor
    scale = max_norm / global_norm

    # 4. Multiply every gradient element by the same factor
    clipped = [[x * scale for x in g] for g in gradients]

    return clipped