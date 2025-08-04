# Standard Library
import abc

# Dependencies
import numpy as np


class Classifier(metaclass=abc.ABCMeta):
    """
    Classifier base abstract class.
    """

    @abc.abstractmethod
    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Trains the classifier.

        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target classes
        """

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output :return (N,) array
            holding the predicted classes
        """
