import numpy as np
from typing import Union
from scipy.special import gamma
from prml.distribution import Distribution


class Dirichlet(Distribution):
    """
    **Dirichlet** distribution:

    p(mu|alpha) = (Γ(a0) / Γ(a1)...Γ(ak)) * prod_k mu_k ^ (a_k - 1)
    """

    def __init__(self, a: np.ndarray):
        """
        Create a *Dirichlet* distribution.

        :param a:
        """
        self.a = a

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
        return gamma(np.sum(self.a)) * np.prod(x ** (self.a - 1)) / np.prod(gamma(self.a))

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        return np.random.dirichlet(alpha=self.a, size=sample_size)
