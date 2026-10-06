import math

def compute_context_compression(
    hist_frames: int,
    height: int,
    width: int,
    latent_channels: int,
    spatial_compression: int,
    temporal_compression: int,
    spatial_downsample: int,
    temporal_downsample: int,
    dtype: str = 'fp16'
) -> dict:
    """
    Compute token counts, memory footprint, and compression ratio for
    historical context compression in an autoregressive video model.
    """

    dtype_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "fp8": 1
    }

    bytes_per_value = dtype_bytes[dtype]

    # Stage 1: VAE compression
    lat_t = math.ceil(hist_frames / temporal_compression)
    lat_h = math.ceil(height / spatial_compression)
    lat_w = math.ceil(width / spatial_compression)

    full_latent_tokens = lat_t * lat_h * lat_w

    full_memory_bytes = (
        full_latent_tokens
        * latent_channels
        * bytes_per_value
    )

    # Stage 2: additional context compression
    comp_t = math.ceil(lat_t / temporal_downsample)
    comp_h = math.ceil(lat_h / spatial_downsample)
    comp_w = math.ceil(lat_w / spatial_downsample)

    compressed_tokens = comp_t * comp_h * comp_w

    compressed_memory_bytes = (
        compressed_tokens
        * latent_channels
        * bytes_per_value
    )

    compression_ratio = full_latent_tokens / compressed_tokens

    memory_saved_bytes = full_memory_bytes - compressed_memory_bytes
    memory_saved_mb = memory_saved_bytes / (1024 ** 2)

    return {
        "full_latent_tokens": full_latent_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": round(compression_ratio, 4),
        "full_memory_bytes": full_memory_bytes,
        "compressed_memory_bytes": compressed_memory_bytes,
        "memory_saved_bytes": memory_saved_bytes,
        "memory_saved_mb": round(memory_saved_mb, 4)
    }