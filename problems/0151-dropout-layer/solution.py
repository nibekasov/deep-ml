import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """
        p: dropout rate = вероятность обнуления нейрона.
           Здесь p трактуем как "drop rate" (обычно 0 <= p < 1).
        """
        assert 0.0 <= p < 1.0, "p must be in [0, 1)"
        self.p = p
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass of the dropout layer.

        x: вход (любая форма)
        training: True — применяем dropout; False — пропускаем без изменений.
        """
        # Режим обучения: генерируем маску и применяем её
        if training and self.p > 0.0:
            keep_prob = 1.0 - self.p
            # Маска: 1 — оставить, 0 — выкинуть
            self.mask = (np.random.rand(*x.shape) < keep_prob).astype(x.dtype)
            # Масштабируем, чтобы сохранялось матожидание
            return x * self.mask / keep_prob

        # Если traini