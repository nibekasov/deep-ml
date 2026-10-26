import numpy as np

def vit_encode(
    image: np.ndarray,
    patch_size: int,
    W_patch: np.ndarray,
    cls_token: np.ndarray,
    pos_embed: np.ndarray,
    W_qkv: np.ndarray,
    W_o: np.ndarray,
    num_heads: int,
    ) -> list[float]:
    """
    Forward pass of a minimal ViT encoder block.

    Returns the final embedding of the CLS token as a list of length D.
    """
    image = np.asarray(image, dtype=float)
    W_patch = np.asarray(W_patch, dtype=float)
    cls_token = np.asarray(cls_token, dtype=float)
    pos_embed = np.asarray(pos_embed, dtype=float)
    W_qkv = np.asarray(W_qkv, dtype=float)
    W_o = np.asarray(W_o, dtype=float)

    H, W, C = image.shape
    p = patch_size
    N = (H // p) * (W // p)
    D = W_patch.shape[1]
    L = N + 1
    d_h = D // num_heads

    
    patches = (
        image.reshape(H // p, p, W // p, p, C)
        .transpose(0, 2, 1, 3, 4)
        .reshape(N, p * p * C)
    )

    # (N, D)
    tokens = patches @ W_patch 

    # (L, D)
    tokens = np.concatenate(
        [cls_token[None, :], tokens], axis=0
    )                                         
    tokens = tokens + pos_embed

    Q, K, V = np.split(tokens @ W_qkv, 3, axis=-1)

    # (L, D) -> (num_heads, L, d_h)
    Q = Q.reshape(L, num_heads, d_h).transpose(1, 0, 2)
    K = K.reshape(L, num_heads, d_h).transpose(1, 0, 2)
    V = V.reshape(L, num_heads, d_h).transpose(1, 0, 2)

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_h)

    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)

    # (num_heads, L, d_h)
    attended = weights @ V                    

    # (L, D)
    attended = attended.transpose(1, 0, 2).reshape(L, D)
    output = attended @ W_o                   

    return output[0].tolist()
