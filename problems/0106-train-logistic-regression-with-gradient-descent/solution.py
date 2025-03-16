import numpy as np


def train_logreg(X: np.ndarray, y: np.ndarray, 
                 learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """        
    Gradient-descent training algorithm for logistic regression, that collects sum-reduced
    BCE losses, accuracies. Assigns label "0" if the P(x_i)<=0.5 and "1" otherwise.

    Returns
    -------
    B : list[float]
        1xM updated parameter vector rounded to 4 floating points
    losses : list[float]
        collected values of a BCE loss function (LLF) rounded to 4 floating points
    """

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def accuracy(y_pred, y_true):
        return (y_true == np.rint(y_pred)).sum() / len(y_true)
    
    def bce_loss(y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    y = y.reshape(-1, 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    B = np.zeros((X.shape[1], 1))
    accuracie