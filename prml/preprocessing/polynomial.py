# Standard Library
import functools
import itertools

# Dependencies
import numpy as np
from numpy.typing import NDArray

# Project
from prml.helpers import array
from prml.preprocessing.basis_function import BasisFunction


class PolynomialFeature(BasisFunction):
    """
    Polynomial basis functions.

    Transforms the input array using polynomial basis functions.

    Example x = [[a, b], [c, d]]

    y = PolynomialBasis(degree=2).transform(x)

    y = [[1, a, b, a^2, a * b, b^2], [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree: int = 2) -> None:
        """
        Create polynomial basis functions.

        :param degree: the degree of polynomial (default is 2)
        """
        if not isinstance(degree, int):
            raise ValueError(f"Degree should be of type 'int', but type '{type(degree)}' is given.")
        self._degree = degree

    def transform(self, x: float | NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Transforms input array using polynomial basis functions.

        :param x: (N, D) array of values, float or int
        :return: (N, 1 + nC1 + ... + nCd) array of polynomial features
        """

        # check if proper array is given or create one if not
        x = array.to_array(x)

        # create a list of ones for the zero powers
        features = [np.ones(len(x))]

        for degree in range(1, self._degree + 1):
            for items in itertools.combinations_with_replacement(x.T, degree):
                features.append(functools.reduce(lambda a, b: a * b, items))

        return np.asarray(features).T


class LinearFeature(PolynomialFeature):
    """
    Linear basis functions.

    Transforms the input array by adding a bias. Identical to

    PolynomialFeature(degree=1)

    Example x = [[a, b], [c, d]]

    y = PolynomialBasis(degree=1).transform(x)

    y = [[1, a, b], [1, c, d]]
    """

    def __init__(self) -> None:
        super().__init__(degree=1)
