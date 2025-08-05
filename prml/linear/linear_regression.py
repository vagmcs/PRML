# Dependencies
import numpy as np

from .regression import Regression


class LinearRegression(Regression):
    """
    Linear regression model.

    y = X @ w t ~ N(t|X @ w, var)
    """

    def __init__(self) -> None:
        """
        Creates a linear regression model.
        """
        self._w: np.ndarray | None = None
        self._var: np.ndarray | None = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Performs the least squares fitting.

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        """

        self._w = np.linalg.pinv(x) @ t
        self._var = np.mean(np.square(x @ self._w - t))

    def fit_lms(self, x: np.ndarray, t: np.ndarray, eta: float = 0.01, n_iter: int = 1000) -> None:
        """
        Stochastic gradient descent using the sum of squares error function is called
        the Least Mean Squares (LMS).

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        :param eta: learning rate
        :param n_iter: number of iterations
        """

        x = x[:, None] if x.ndim == 1 else x
        n, d = x.shape
        indices = np.arange(n)

        self._w = np.random.random(d)
        for _ in range(n_iter):
            # shuffle the data
            indices = np.random.permutation(indices)
            for i in indices:
                self._w = self._w + eta * (t[i] - np.dot(self._w.T, x[i])) * x[i]

        self._var = np.mean(np.square(x @ self._w - t))

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output :return a tuple of
            (N,) arrays, one holding the predictions, and one the variance
        """

        if self._w is None or self._var is None:
            raise ValueError("The model is not trained, thus predictions cannot be made!")

        y = x @ self._w
        y_std = np.sqrt(self._var) + np.zeros_like(y)
        return y, y_std
