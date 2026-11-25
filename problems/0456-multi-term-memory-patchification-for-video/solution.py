import math

def multi_term_memory_patchification(
    terms, latent_h, latent_w
):
    tokens_per_term = []

    for term in terms:
        # Temporal downsampling
        t = math.ceil(
            term["num_latent_frames"] / term["temporal_stride"]
        )

        # Spatial downsampling
        h = math.ceil(latent_h / term["spatial_stride"])
        w = math.ceil(latent_w / term["spatial_stride"])

        # Patchification
        ph = math.ceil(h / term["patch_size"])
        pw = math.ceil(w / term["patch_size"])

        tokens_per_term.append(t * ph * pw)

    total = sum(tokens_per_term)

    return {
        "tokens_per_term": tokens_per_term,
        "total_tokens": total,
        "token_fractions": [
            round(n / total, 4) if total else 0.0
            for n in tokens_per_term
        ]
    }