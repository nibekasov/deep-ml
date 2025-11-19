import math
import torch
from torch.optim.optimizer import Optimizer

class MyOptimizer(Optimizer):
    """
    Простой адаптивный оптимизатор в стиле Adam.
    - Поддерживает плотные градиенты.
    - Хорошо работает на MNIST для маленьких CNN.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """
        params       : параметры модели (iterable)
        lr           : базовый learning rate
        betas        : (beta1, beta2) — коэффициенты экспоненциального сглаживания
        eps          : численная стабилизация в знаменателе
        weight_decay : L2-регуляризация (0.0 — выключена)
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Один шаг оптимизации.
        closure (опционально): функция, которая пересчитывает loss.
        Возвращает loss, если closure задан.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # Только плотные градиенты
                if grad.is_sparse:
                    raise RuntimeError("MyOptimizer does not support sparse gradients")

                # Добавим L2-регуляризацию (если включена)
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # Инициализация state для конкретного параметра
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Первый момент (m_t)
                    state["exp_avg"] = torch.zeros_like(p)
                    # Второй момент (v_t)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Обновление первого момента: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Обновление второго момента: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias-correction для моментов:
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                # Нормированная оценка дисперсии
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Эффективный lr с учётом bias-correction первого момента
                step_size = lr / bias_correction1

                # Обновление параметров:
                # p = p - step_size * m_hat / (sqrt(v_hat) + eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
