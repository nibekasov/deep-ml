import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    n_batch, l_seq, d_model = x.shape
    assert We.shape == (n_experts, d_model, d_model)
    assert Wg.shape == (d_model, n_experts)
    assert 1 <= top_k <= n_experts

    # (n_batch * l_seq, d_model)
    N = n_batch * l_seq
    x_flat = x.reshape(N, d_model).astype(np.float64)

    # 1) Гейтинг: logits -> softmax по экспертам
    logits = x_flat @ Wg                      # (N, n_experts)
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - log