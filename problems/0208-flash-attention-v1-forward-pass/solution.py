import numpy as np

def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                            block_size: int = 2) -> np.ndarray:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix of shape (seq_len, d_model)
        K: Key matrix of shape (seq_len, d_model)
        V: Value matrix of shape (seq_len, d_model)
        block_size: Tile size for blocked computation along sequence axis.
    
    Returns:
        Output matrix of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    scale = 1.0 / np.sqrt(d_model)

    # Результат
    out = np.zeros_like(V, dtype=np.float64)

    # Идём по блокам Q
    for i in range(0, seq_len, block_size):
        bq = min(block_size, seq_len - i)    # реальный размер блока по Q
        Q_block = Q[i:i + bq]               # (bq, d_model)

        # Онлайн-статистики для этого блока Q
        m = np.full((bq,), -np.inf, dtype=np