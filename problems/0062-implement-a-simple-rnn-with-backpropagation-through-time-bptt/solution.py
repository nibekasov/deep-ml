
import numpy as np
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
		"""
		Initializes the RNN with random weights and zero biases.
		"""
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size)*0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
        self.W_hy = np.random.randn(output_size, hidden_size)*0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
    def forward(self, x):
        """
        Forward pass through the RNN for a given sequence of inputs.
        x: np.ndarray of shape (T, input_size)
        Returns: y_preds of shape (T, output_size)
        """
        x = np.asarray(x, dtype=float)
        T = x.shape[0]
        h_prev = np.zeros((self.hidden_size, 1))

        # Кэши для BPTT
        self._xs = []
        self._hs = [h_prev]   # h_0
        self._ys = []

        for t in range(T):