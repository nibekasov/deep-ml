import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    d_k = K.shape[-1]  # Scaling factor
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Apply the mask (add a large negative value to masked positions)
    scores += mask
    
    # Ensure numerical stability
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Compute the final output
    return np.dot(attention_weights, V)

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute self-attention mechanism.
    """
    d_k =