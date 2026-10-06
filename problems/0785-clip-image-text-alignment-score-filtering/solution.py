import numpy as np

def clip_score_filter(image_embeds, text_embeds, threshold: float):
    """
    Compute pairwise cosine similarities between image and text embeddings
    (assumed L2-normalized) and zero out entries below `threshold`.

    Returns:
        Nested list of shape (n_images, n_texts).
    """
    image_embeds = np.asarray(image_embeds, dtype=float)
    text_embeds = np.asarray(text_embeds, dtype=float)

    if len(image_embeds) == 0 or len(text_embeds) == 0:
        return []

    # Pairwise cosine similarity
    similarities = image_embeds @ text_embeds.T

    # Zero out values strictly below threshold
    similarities[similarities < threshold] = 0.0

    return similarities.tolist()