# Standard Library
import abc

# Dependencies
import numpy as np
from numpy.typing import NDArray


class BasisFunction(abc.ABC):
    """
    Basis function abstract class.
    """

    @abc.abstractmethod
    def transform(self, x: float | NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Transforms input array using the basis functions.

        :param x: (N, D) array of values, float or int
        :return: an array of features
        """
