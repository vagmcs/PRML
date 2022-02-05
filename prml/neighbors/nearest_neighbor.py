# Types
from typing import Union

# Dependencies
import numpy as np
from scipy.spatial import distance_matrix

# Project
from .neighbors import Neighbors


class KNearestNeighborsClassifier(Neighbors):
    """
    K-nearest neighbor classifier.
    """

    def __init__(self, k: int, x: Union[int, float, np.ndarray], t: Union[int, float, np.ndarray]):
        """
        Creates a nearest neighbor classifier.

        :param k: number of nearest neighbors
        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target values
        """
        super().__init__(k, x)

        if isinstance(t, (int, float)):
            self._t = np.array([t])
        elif isinstance(t, np.ndarray):
            if t.ndim > 1:
                raise ValueError("Target values should be an (N,) 1D array, where N is the number of targets.")
            self._t = t[:, None]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output
        :return (N,) array holding the probabilistic predictions
        """
        d = distance_matrix(x, self._data)
        indices = d.argsort()[:, : self._k]
        counts_per_class = np.apply_along_axis(lambda row: np.bincount(row, minlength=2), axis=1, arr=self._t[indices])
        return counts_per_class / self._k

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output
        :return (N,) array holding the predictions
        """
        d = distance_matrix(x, self._data)
        indices = d.argsort()[:, : self._k]
        # count the neighbors from each target (in the k-neighborhood)
        counts_per_class = np.apply_along_axis(lambda row: np.bincount(row, minlength=2), axis=1, arr=self._t[indices])
        return counts_per_class.argmax(axis=1)  # type: ignore
