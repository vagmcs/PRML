# Types
from typing import Union

# Standard Library
import abc

# Dependencies
import numpy as np


class Neighbors(metaclass=abc.ABCMeta):
    """
    Nearest neighbors base abstract class.
    """

    def __init__(self, k: int, data: Union[int, float, np.ndarray]):
        """
        :param k: number of nearest neighbors
        :param data: (N, D) array holding the input training data
        """
        self._k = k

        if isinstance(data, (int, float)):
            self._data = np.array([[data]])
        elif isinstance(data, np.ndarray):
            if data.ndim > 2:
                raise ValueError(
                    "Input data should be an (N, D) 2D array, where N is the number of samples "
                    "and D is the dimension of each sample."
                )
            self._data = data[:, None] if data.ndim == 1 else data

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output :return (N,) array
            holding the predictions
        """
