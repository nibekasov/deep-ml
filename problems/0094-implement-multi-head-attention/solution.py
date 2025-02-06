import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
	Q = np.dot(X, W_q)
	K = np.dot(X, W_k)
	V = np.dot(X, W_v)
	return Q, K, V

def self_attention(Q, K, V):
	d_k = Q.shape[-1]
	scores = np.dot(Q, K.T) / np.sqrt(d_k)
	attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
	attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
	return np.dot(attention_weights, V)

def multi_head_attention(Q, K, V, n_heads):
	d_model = Q.shape[-1]
	d_k = d_model // n_heads

	Q_split = np.split(Q, n_heads, axis=-1)
	K_split = np.split(K, n_heads, axis=-1)
	V_split = np.split(V, n_heads, axis=-1)

	head_outputs = []
	for i in range(n_heads):
		head_output = self_attention(Q_split[i], K_split[i], V_split[i])
		head_outputs.append(head_output)

	concatenated = np.concatenate(head_outputs, axis=-1)
	return concatenated
