import torch

class MyTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 28, 28) float tensor in [0,1]
        Return: transformed tensor, same shape/dtype.
        """

        # Пер-изображенческая нормализация: (x - mean) / std
        # mean и std — скаляры для конкретного примера
        mean = x.mean()
        std = x.std()

        # Защита от нулевого std
        eps = 1e-6
        std = torch.clamp(std, min=eps)

        x = (x - mean) / std

        # Ограничим значения, чтобы не было слишком больших выбросов
        x = x.clamp(min=-3.0, max=3.0)

        # Форма и dtype сохранены
        return x
