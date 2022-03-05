# Types
from typing import Union

# Dependencies
import numpy as np
from scipy.spatial import distance_matrix
from scipy.special import gamma

# Project
from .neighbors import Neighbors


class NearestNeighborsDensity(Neighbors):
    """
    Nearest neighbour method for density estimation.
    """

    def __init__(self, k: int, data: Union[int, float, np.ndarray]):
        """
        Creates a nearest neighbor estimator.

        :param k: number of nearest neighbors
        :param data: (N, D) array holding the input training data
        """
        super().__init__(k, data)
        self._D = data.shape[1] if isinstance(data, np.ndarray) and data.ndim != 1 else 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output
        :return (N,) array holding the predictions
        """
        d = distance_matrix(x.reshape(-1, 1), self._data.reshape(-1, 1))
        radius = np.apply_along_axis(np.sort, 1, d)[:, self._k - 1]
        volume = (np.pi ** (self._D / 2) * radius**self._D) / gamma(self._D / 2 + 1)
        return self._k / (volume * self._data.size)  # type: ignore
