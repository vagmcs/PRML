# Types
from typing import Optional, Tuple, Union

# Dependencies
import numpy as np

from .regression import Regression


class BayesianRegression(Regression):
    """
    Bayesian regression model.

    w ~ N(w|0, alpha^(-1)I) y(x, w) = w.T * X t ~ N(t|y(x, w), beta^(-1))
    """

    def __init__(self, alpha: Union[int, float], beta: Union[int, float]):
        self._alpha = alpha
        self._beta = beta
        self._mean: Optional[np.ndarray] = None
        self._precision: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Trains the model by estimating the posterior distribution (3.49).

        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target values
        """

        # assume zero-mean, isotropic Gaussian prior
        if self._mean is None or self._precision is None:
            self._mean = np.zeros(x.shape[1])
            self._precision = self._alpha * np.eye(x.shape[1])

        mean_prev, precision_prev = self._mean, self._precision

        self._precision = precision_prev + self._beta * x.T @ x
        self._cov = np.linalg.inv(self._precision)  # type: ignore

        self._mean = np.linalg.solve(self._precision, precision_prev @ mean_prev + self._beta * x.T @ t)  # type: ignore

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output :return a tuple of
            (N,) arrays, one holding the predictions, and one the variance
        """
        if self._mean is None or self._cov is None:
            raise ValueError("The model is not trained, thus predictions cannot be made!")

        # the maximum posterior weight vector is simply the mean of the distribution
        y = x @ self._mean
        y_std = np.sqrt(1 / self._beta + np.sum(x @ self._cov * x, axis=1))
        return y, y_std

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the underlying Gaussian distribution.

        :param sample_size: number of samples
        :return: an array holding the drawn samples
        """
        return np.random.multivariate_normal(self._mean, self._cov, size=sample_size)  # type: ignore
