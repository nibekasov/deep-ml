import numpy as np

def gambler_value_iteration(ph, theta=1e-9):
    """
    Computes the optimal value function and policy for the Gambler's Problem.
    Uses: reward=0 everywhere, terminal V[100]=1 (probability of success).
    """
    ph = float(ph)
    V = np.zeros(101, dtype=np.float64)
    V[100] = 1.0  # goal value = 1 (probability of eventual success)

    # Value iteration
    while True:
        delta = 0.0
        for s in range(1, 100):
            old = V[s]
            max_stake = min(s, 100 - s)
            # Evaluate all stakes a in [1, max_stake]
            vals = []
            for a in range(1, max_stake + 1):
                # reward = 0; just expected next-state value
                vals.append(ph * V[s + a] + (1.0 - ph) * V[s - a])
            V[s] = max(vals) if vals else 0.0
            delta = max(delta, abs(old - V[s]))
        if delta < theta:
            break

    # Greedy policy extraction (any argmax on ties)
    policy = [0