# Types
from typing import Union

# Dependencies
import numpy as np

# Project
from .regression import Regression


class BayesianRegression(Regression):
    """
    Bayesian regression model

    w ~ N(w|0, alpha^(-1)I)
    y(x, w) = w.T * X
    t ~ N(t|y(x, w), beta^(-1))
    """

    def __init__(self, alpha: Union[int, float], beta: Union[int, float]):
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.precision = None
        self.cov = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """

        TODO: Each time fit is called the prior knowledge is retained!
        """

        if self.mean is None and self.precision is None:
            self.mean = np.zeros(x.shape[1])
            self.precision = self.alpha * np.eye(x.shape[1])

        mean_prev, precision_prev = self.mean, self.precision

        self.precision = precision_prev + self.beta * x.T @ x
        self.cov = np.linalg.inv(self.precision)

        self.mean = np.linalg.solve(self.precision, precision_prev @ mean_prev + self.beta * x.T @ t)

    def predict(self, x: np.ndarray, return_std: bool = False):
        """ """
        y = x @ self.mean

        if return_std:
            y_var = 1 / self.beta + np.sum(x @ self.cov * x, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y

    def draw(self, sample_size: int) -> np.ndarray:
        """ """
        return np.random.multivariate_normal(self.mean, self.cov, size=sample_size)
