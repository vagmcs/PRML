# Dependencies
import numpy as np

from .regression import Regression


class RidgeRegression(Regression):
    """
    Ridge regression model.

    w* = argmin |t - X @ w| + lambda * |w|_2^2
    """

    def __init__(self, alpha: float = 1):
        self._w: np.ndarray | None = None
        self._var: np.ndarray | None = None
        self._alpha = alpha

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Maximum a posteriori estimation of parameter.

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        """

        eye = np.eye(np.size(x, 1))
        self._w = np.linalg.pinv(x.T @ x + self._alpha * eye) @ x.T @ t
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
