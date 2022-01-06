# Types
from typing import Union

# Standard Library
import abc

# Dependencies
import numpy as np


class BasisFunction(metaclass=abc.ABCMeta):
    """
    Basis function abstract class.
    """

    @abc.abstractmethod
    def transform(self, x: Union[int, float, np.ndarray]) -> np.ndarray:
        """
        Transforms input array using the basis functions.

        :param x: (N, D) array of values, float or int
        :return: an array of features
        """
