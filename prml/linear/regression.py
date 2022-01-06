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

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        """

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) numpy array sample to predict their output
        :return (N,) numpy array holding the prediction of each input
        """
