import abc
import numpy as np


class Regression(metaclass=abc.ABCMeta):
    """
    Regression is an abstract class that should be extended
    by any algorithm for early time-series classification.
    """

    @abc.abstractmethod
    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Trains the classifier.

        :param x: training time-series data
        :param t: training time-series labels
        """

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of the given time-series as early as possible.

        :param x: time-series to predict
        :return
        """
