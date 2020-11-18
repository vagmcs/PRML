import numpy as np
from typing import Union
from scipy.special import gamma
from prml.distribution import Distribution


class Beta(Distribution):
    """
    **Beta** distribution:

    p(mu|a, b) = (Γ(a + b) / Γ(a)Γ(b)) * mu^(a - 1) * (1 - mu)^(b - 1)
    """

    def __init__(self, a: float, b: float):
        """
        Create a *Beta* distribution.

        :param a: number of ones
        :param b: number of zeros
        """
        self.a = a
        self.b = b

    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        """
        pass

    def pdf(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        return (
            gamma(self.a + self.b) / (gamma(self.a) * gamma(self.b))
            * np.power(x, self.a - 1) * np.power(1 - x, self.b - 1)
        )

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        return np.random.beta(a=self.a, b=self.b, size=sample_size)
