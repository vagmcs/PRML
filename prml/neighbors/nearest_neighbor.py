# Dependencies
import numpy as np
from scipy.spatial import distance_matrix


class KNearestNeighborsClassifier:
    """
    Regression base abstract class.
    """

    def __init__(self, k):
        self.k = k
        self.x = None
        self.t = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Trains the model.

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        """
        self.x = x
        self.t = t

    def probability(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) numpy array sample to predict their output
        :return (N,) numpy array holding the prediction of each input
        """
        d = distance_matrix(x, self.x)
        indices = d.argsort()[:, : self.k]
        counts_per_class = np.apply_along_axis(lambda row: np.bincount(row, minlength=2), axis=1, arr=self.t[indices])
        return counts_per_class / self.k

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) numpy array sample to predict their output
        :return (N,) numpy array holding the prediction of each input
        """
        d = distance_matrix(x, self.x)
        indices = d.argsort()[:, : self.k]
        return np.apply_along_axis(lambda row: np.bincount(row, minlength=2), axis=1, arr=self.t[indices]).argmax(
            axis=1
        )
