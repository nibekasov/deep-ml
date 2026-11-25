import numpy as np

def video_tokenizer_reconstruct(video, patch_t: int, patch_h: int, patch_w: int, mode: str):
    video = np.asarray(video)
    T, H, W, C = video.shape
    pt = 1 if mode == "spatial" else patch_t

    patches = video.reshape(
        T // pt, pt,
        H // patch_h, patch_h,
        W // patch_w, patch_w,
        C,
    )
    means = patches.mean(axis=(1, 3, 5), keepdims=True)
    return np.broadcast_to(means, patches.shape).reshape(video.shape).tolist()
