import numpy as np

def pos_encoding(position: int, d_model: int):
	if position <= 0 or d_model <= 0:
		return -1

	PE = np.zeros((position, d_model), dtype=np.float16)

	div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

	for pos in range(position):
		PE[pos, 0::2] = np.sin(pos * div_term)
		PE[pos, 1::2] = np.cos(pos * div_term)

	return PE[np.newaxis, :, :].astype(np.float16)