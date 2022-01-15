# Dependencies
import numpy as np

# Project
from .regression import Regression


class LinearRegression(Regression):
    """
    Linear regression model

    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def __init__(self):
        self.w = None
        self.var = None

    def fit(self, x: np.ndarray, t: np.ndarray):
        """
        Performs the least squares fitting

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        """

        self.w = np.linalg.pinv(x) @ t
        self.var = np.mean(np.square(x @ self.w - t))

    def fit_lms(self, x: np.ndarray, t: np.ndarray, eta: float = 0.01, n_iter: int = 1000):
        """
        Stochastic gradient descent using the sum of squares error
        function is called the Least Mean Squares (LMS).

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        :param eta: learning rate
        :param n_iter: number of iterations
        """

        x = x[:, None] if x.ndim == 1 else x
        n, d = x.shape
        indices = np.arange(n)

        self.w = np.random.random(d)
        for _ in range(n_iter):
            indices = np.random.permutation(indices)  # shuffle the data
            for i in indices:
                self.w = self.w + eta * (t[i] - np.dot(self.w.T, x[i])) * x[i]

    def predict(self, x: np.ndarray, return_std: bool = False):
        """
        Makes a prediction given an input.

        :param x: (N, D) numpy array sample to predict their output
        :param return_std: bool, optional returns standard deviation of each prediction if True
        :return:
        (N,) numpy array holding the prediction of each input, and
        (N,) numpy array holding the standard deviation of each prediction
        """

        y = x @ self.w
        if return_std and self.var is not None:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y
