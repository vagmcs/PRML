import numpy as np
from scipy.special import factorial
from typing import Union
from prml.distribution.distribution import Distribution


class Binomial(Distribution):
    """
    **Binomial** distribution:

    p(m|N,mu) = (N!/(N - m)!m!) * mu^m * (1 - mu)^(N - m)
    """

    def __init__(self, n: int, mu: float = 0.5):
        """
        Create a *Binomial* distribution. By default the distribution
        models a fair coin toss.

        :param n: number of trials
        :param mu: the probability of the binary random variable to be True (default is 0.5)
        """
        self.n = n
        self.mu = mu

    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        """
        self.mu = np.mean(x)

    def pdf(self, x: Union[np.ndarray, int]) -> Union[np.ndarray, int]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        binomial_coefficient = factorial(self.n) / (factorial(x) * factorial(self.n - x))
        return binomial_coefficient * (self.mu ** x) * (1 - self.mu) ** (self.n - x)

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        return np.random.binomial(n=self.n, p=self.mu, size=sample_size)
