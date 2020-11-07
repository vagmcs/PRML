import numpy as np
from prml.linear.regression import Regression


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
        Performs least squares fitting

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        """

        self.w = np.linalg.pinv(x) @ t
        self.var = np.mean(np.square(x @ self.w - t))

    def predict(self, x: np.ndarray, return_std: bool = False):
        """
        Makes a prediction given an input.

        :param x: (N, D) numpy array sample to predict their output
        :param return_std: bool, optional returns standard deviation of each prediction if True
        :return:
        (N,) numpy array holding the prediction of each input
        (N,) numpy array holding the standard deviation of each prediction
        """

        y = x @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y
