import numpy as np

def q_learning(num_states, num_actions, P, R, terminal_states,
               alpha, gamma, epsilon, num_episodes):
    """
    Learn Q-table with Q-Learning on a tabular MDP.

    Args:
        num_states (int)
        num_actions (int)
        P (np.ndarray): shape (S, A, S), transition probabilities.
        R (np.ndarray): shape (S, A), immediate rewards.
        terminal_states (list or np.ndarray): terminal state indices.
        alpha (float): learning rate in [0,1].
        gamma (float): discount factor in [0,1].
        epsilon (float): epsilon-greedy exploration in [0,1].
        num_episodes (int): number of episodes (>=1).

    Returns:
        np.ndarray: Q-table of shape (S, A).
    """
    S, A = int(num_states), int(num_actions)
    Q = np.zeros((S, A), dtype=float)

    terminal = np.zeros(S, dtype=bool)
    terminal[np.asarray(terminal_states, dtype=int)] = True
    non_terminal_states = np.where(~terminal)[0]
    if non