import numpy as np

def video_tokenize(video, codebook, encoder_weight, patch_size):
    try:
        video = np.asarray(video)
        codebook = np.asarray(codebook)
        encoder_weight = np.asarray(encoder_weight)
        pt, ph, pw = patch_size
    except (TypeError, ValueError):
        return -1

    for size in (pt, ph, pw):
        if isinstance(size, (bool, np.bool_)):
            return -1
        if not isinstance(size, (int, np.integer)) or size <= 0:
            return -1

    if video.ndim != 4 or codebook.ndim != 2 or encoder_weight.ndim != 2:
        return -1

    T, H, W, C = video.shape
    K, D = codebook.shape
    if min(T, H, W, C, K, D) == 0:
        return -1
    if T % pt or H % ph or W % pw:
        return -1
    if encoder_weight.shape != (pt * ph * pw * C, D):
        return -1

    for array in (video, codebook, encoder_weight):
        if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            return -1

    nt, nh, nw = T // pt, H // ph, W // pw
    patches = (
        video.reshape(nt, pt, nh, ph, nw, pw, C)
        .transpose(0, 2, 4, 1, 3, 5, 6)
        .reshape(nt * nh * nw, pt * ph * pw * C)
    )

    embeddings = patches.astype(np.float64) @ encoder_weight
    codebook = codebook.astype(np.float64)

    indices = []
    for embedding in embeddings:
        distances_sq = np.sum((codebook - embedding) ** 2, axis=1)
        indices.append(np.argmin(distances_sq))

    return np.asarray(indices).reshape(nt, nh, nw).tolist()