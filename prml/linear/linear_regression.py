import numpy as np
from prml.linear.regression import Regression


class LinearRegression(Regression):
    """
    Linear regression model
    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def fit(self, x: np.ndarray, t: np.ndarray):
        """
        perform least squares fitting
        Parameters
        ----------
        x : (N, D) np.ndarray
            training independent variable
        t : (N,) np.ndarray
            training dependent variable
        """
        self.w = np.linalg.pinv(x) @ t
        self.var = np.mean(np.square(x @ self.w - t))

    def predict(self, x: np.ndarray, return_std: bool = False):
        """
        make prediction given input
        Parameters
        ----------
        x : (N, D) np.ndarray
            samples to predict their output
        return_std : bool, optional
            returns standard deviation of each predition if True
        Returns
        -------
        y : (N,) np.ndarray
            prediction of each sample
        y_std : (N,) np.ndarray
            standard deviation of each predition
        """
        y = x @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y
