# Types
from typing import Tuple

# Standard Library
import abc

# Dependencies
import numpy as np


class Regression(metaclass=abc.ABCMeta):
    """
    Regression base abstract class.
    """

    @abc.abstractmethod
    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Trains the model.

        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target values
        """

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output
        :return a tuple of (N,) arrays, one holding the predictions, and one the variance
        """
