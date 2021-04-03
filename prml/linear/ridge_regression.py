import numpy as np

from .regression import Regression


class RidgeRegression(Regression):
    """
    Ridge regression model

    w* = argmin |t - X @ w| + lambda * |w|_2^2
    """

    def __init__(self, alpha: float = 1):
        self.w = None
        self.alpha = alpha

    def fit(self, x: np.ndarray, t: np.ndarray):
        """
        Maximum a posteriori estimation of parameter

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        """

        eye = np.eye(np.size(x, 1))
        self.w = np.linalg.pinv(x.T @ x + self.alpha * eye) @ x.T @ t

    def predict(self, x: np.ndarray):
        """
        Makes a prediction given an input.

        :param x: (N, D) numpy array sample to predict their output
        :return: (N,) numpy array holding the prediction of each input
        """

        return x @ self.w
