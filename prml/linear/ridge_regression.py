import numpy as np
from prml.linear.regression import Regression


class RidgeRegression(Regression):
    """
    Ridge regression model

    w* = argmin |t - X @ w| + alpha * |w|_2^2
    """

    def __init__(self, alpha: float = 1):
        self.alpha = alpha

    def fit(self, x: np.ndarray, t: np.ndarray):
        """
        maximum a posteriori estimation of parameter

        Parameters
        ----------
        x : (N, D) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
        """

        eye = np.eye(np.size(x, 1))
        self.w = np.linalg.solve(self.alpha * eye + x.T @ x, x.T @ t)

    def predict(self, x: np.ndarray):
        """
        make prediction given input

        Parameters
        ----------
        x : (N, D) np.ndarray
            samples to predict their output

        Returns
        -------
        (N,) np.ndarray
            prediction of each input
        """
        return x @ self.w
