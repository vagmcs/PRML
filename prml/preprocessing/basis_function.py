import abc
import numpy as np
from typing import Union


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