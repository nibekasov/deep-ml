import numpy as np

def rnn_forward(input_sequence: list[list[float]],
			    initial_hidden_state: list[float],
				Wx: list[list[float]],
				Wh: list[list[float]],
				b: list[float]) -> list[float]:
	h = np.array(initial_hidden_state)
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	b = np.array(b)
	for x in input_sequence:
		x = np.array(x)
		h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)
	final_hidden_state = h
	return final_hidden_state

	return final_hidden_state