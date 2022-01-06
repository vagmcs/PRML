# Types
from typing import Union

# Standard Library
import abc

# Dependencies
import numpy as np
from scipy.spatial import distance_matrix
from scipy.special import gamma


class NearestNeighborsDensity:
    """
    Regression base abstract class.
    """

    def __init__(self, k):
        self.k = k
        self.x = None
        self.D = None

    def fit(self, x: Union[float, np.ndarray]) -> None:
        """
        Trains the model.

        :param x: (N, D) numpy array holding the input training data
        """
        self.x = x
        self.D = x.shape[1] if isinstance(x, np.ndarray) and x.ndim != 1 else 1

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) numpy array sample to predict their output
        :return (N,) numpy array holding the prediction of each input
        """
        d = distance_matrix(x.reshape(-1, 1), self.x.reshape(-1, 1))
        radius = np.apply_along_axis(np.sort, 1, d)[:, self.k - 1]
        volume = (np.pi ** (self.D / 2) * radius ** self.D) / gamma(self.D / 2 + 1)
        return self.k / (volume * self.x.size)
