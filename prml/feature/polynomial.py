import itertools
import functools
import numpy as np


class PolynomialFeature(object):
    """
    Polynomial features

    transforms input array with polynomial features

    Example
    =======
    x =
    [[a, b],
    [c, d]]

    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree: int = 2):
        """
        Create polynomial features

        :param degree: the degree of polynomial
        """

        self.degree = degree

    def transform(self, x):
        """
        Transforms input array using polynomial features

        :param x: (sample_size, n) numpy array
        :return: (sample_size, 1 + nC1 + ... + nCd) numpy array of polynomial features
        """

        # Proper shape for 1-dimensional vectors
        x = x[:, None] if x.ndim == 1 else x

        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x.T, degree):
                features.append(functools.reduce(lambda a, b: a * b, items))
        return np.asarray(features).T
