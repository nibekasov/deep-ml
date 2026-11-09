import numpy as np

def sample_and_preprocess(video, num_frames: int) -> list:
    video = np.asarray(video)
    indices = np.rint(
        np.linspace(0, video.shape[0] - 1, num_frames)
    ).astype(int)

    frames = video[indices].astype(float) / 255.0
    return frames.tolist()