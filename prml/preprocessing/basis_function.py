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

    @staticmethod
    def _make_array(x: Union[int, float, np.ndarray]) -> np.ndarray:
        """
        Checks if the input is an array. The array should be at most
        2-dimensional, representing one data point per row. If not, an
        exception is raised. In case a single number is given, it creates
        an array containing a single number.

        :param x: (N, D) array of values or float or int
        :return: a properly shaped (N, D) array
        """
        if isinstance(x, np.ndarray):
            if x.ndim > 2:
                raise ValueError(
                    "Input data should be an (N, D) array, where N is the number of samples "
                    "and D is the dimension of each sample."
                )
            return x[:, None] if x.ndim == 1 else x  # create proper shape for 1-dimensional arrays
        elif isinstance(x, (int, float)):
            return np.array([[x]])
        else:
            raise ValueError(f"Incompatible type '{type(x)}'.")

    @abc.abstractmethod
    def transform(self, x: Union[int, float, np.ndarray]) -> np.ndarray:
        """
        Transforms input array using the basis functions.

        :param x: (N, D) array of values, float or int
        :return: an array of features
        """
