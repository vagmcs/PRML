import itertools
import functools
import numpy as np
from typing import Union
from prml.feature.basis_function import BasisFunction


class PolynomialBasis(BasisFunction):
    """
    Polynomial basis functions.

    Transforms the input array using polynomial basis functions.

    Example
    =======
    x = [[a, b], [c, d]]

    y = PolynomialFeatures(degree=2).transform(x)

    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree: int = 2):
        """
        Create polynomial basis functions

        :param degree: the degree of polynomial (default is 2)
        """

        assert isinstance(degree, int), f"Degree should be of type 'int', but type '{type(degree)}' was found."
        self.degree = degree

    def transform(self, x: Union[int, float, np.ndarray]) -> np.ndarray:
        """
        Transforms input array using polynomial basis functions

        :param x: (N, D) array of values, float or int
        :return: (N, 1 + nC1 + ... + nCd) array of polynomial features
        """

        # Proper shape for 1-dimensional vectors
        if isinstance(x, np.ndarray):
            x = x[:, None] if x.ndim == 1 else x
        elif isinstance(x, int) or isinstance(x, float):
            x = np.array([[x]])
        else:
            raise ValueError(f'Incompatible type {type(x)}.')

        features = [np.ones(len(x))]  # create a list of ones for the zero powers
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x.T, degree):
                features.append(functools.reduce(lambda a, b: a * b, items))

        return np.asarray(features).T
