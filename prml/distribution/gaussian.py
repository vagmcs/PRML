import math
import numpy as np
from typing import Union
from prml.distribution import Distribution


class Gaussian(Distribution):
    """
    **Gaussian** distribution:

    p(x|mu,var) = exp{-0.5 * (x - mu)^2 / var} / sqrt(2pi * var)
    """

    def __init__(self, mu: float = 0, var: float = 1):
        """

        :param mu:
        :param var:
        """
        self.mu = mu
        self.var = var

    def ml(self, x: np.ndarray, unbiased: bool = False) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        :param unbiased:
        """
        self.mu = np.mean(x)
        if unbiased:
            self.var = np.sum(np.power(x - self.mu, 2)) / (x.size - 1)
        else:
            self.var = np.var(x)

    def pdf(self, x: Union[np.ndarray, int]) -> Union[np.ndarray, int]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        d = x - self.mu
        return np.exp(-d ** 2 / (2 * self.var)) / np.sqrt(2 * np.pi * self.var)

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        return np.random.normal(loc=self.mu, scale=math.sqrt(self.var), size=sample_size)
