import numpy as np

def guidance_attention_mask(chunk_sizes, current_chunk):
    chunk_ids = np.repeat(
        np.arange(len(chunk_sizes)), chunk_sizes
    )

    row = chunk_ids[:, None]
    col = chunk_ids[None, :]

    pos = np.arange(len(chunk_ids))

    # Исторические токены: bidirectional attention
    history = (row < current_chunk) & (col < current_chunk)

    # Текущий чанк: causal attention
    current = (row == current_chunk) & (
        pos[None, :] <= pos[:, None]
    )

    return history | current