import numpy as np
from typing import Union
from prml.distribution import Distribution


class Bernoulli(Distribution):
    """
    **Bernoulli** distribution:

    p(x|mu) = mu^x * (1 - mu)^(x - 1)
    """

    def __init__(self, mu: float = 0.5):
        """
        Create a *Bernoulli* distribution. By default the distribution
        models a fair coin toss.

        :param mu: the probability of the binary random variable to be True (default is 0.5)
        """
        self.mu = mu

    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        """
        # The maximum likelihood estimator for mu parameter in a Bernoulli
        # distribution is the sample mean.
        self.mu = np.mean(x)

    def pdf(self, x: Union[np.ndarray, int]) -> Union[np.ndarray, int]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        return self.mu ** x * (1 - self.mu) ** (1 - x)

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        return (self.mu > np.random.uniform(size=sample_size)).astype(int)
