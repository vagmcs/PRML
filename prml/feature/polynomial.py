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
        Construct polynomial features

        :param degree: the degree of polynomial
        """

        self.degree = degree

    def transform(self, x):
        """
        Transforms input array with polynomial features

        :param x: (sample_size, n) numpy array
        :return: (sample_size, 1 + nC1 + ... + nCd) numpy array of polynomial features
        """

        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda a, b: a * b, items))
        return np.asarray(features).transpose()
