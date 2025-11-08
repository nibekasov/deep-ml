import numpy as np

class GradientBandit:
    def __init__(self, num_actions, alpha=0.1):
        self.num_actions = int(num_actions)
        self.alpha = float(alpha)
        self.preferences = np.zeros(self.num_actions, dtype=float)
        self.avg_reward = 0.0
        self.time = 0

    def softmax(self):
        z = self.preferences - np.max(self.preferences)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def select_action(self):
        probs = self.softmax()
        return int(np.random.choice(self.num_actions, p=probs))

    def update(self, action, reward):
        probs = self.softmax()

        # Обновляем baseline СНАЧАЛА (включая текущую награду) — как в ожидаемом тесте
        self.time += 1
        self.avg_reward += (reward - self.avg_reward) / self.time

        # Затем считаем delta относительно уже обновлённого baseline
        delta = reward - self.avg_reward

        # Градиентное обновление предпочтений
        on