# Dependencies
import numpy as np

# Project
from prml.preprocessing import OneHotEncoder

from .classifier import Classifier


class LeastSquaresClassifier(Classifier):
    """
    Least squares classifier.

    y = argmax_k X @ W_k
    """

    def __init__(self) -> None:
        """
        Creates a least squares classifier.
        """
        self._w: np.ndarray | None = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Trains the classifier.

        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target classes
        """

        t_one_hot = OneHotEncoder.encode(t)
        self._w = np.linalg.pinv(x) @ t_one_hot

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output :return (N,) array
            holding the predicted classes
        """
        return np.argmax(x @ self._w, axis=-1)  # type: ignore
