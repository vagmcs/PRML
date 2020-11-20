import numpy as np
import sympy as sym
import sympy.abc as symbols
from typing import Union, Optional
from scipy.special import gamma
from prml.distribution import GenericDistribution


class Dirichlet(GenericDistribution):
    """
    The **Dirichlet** distribution:

    p(x|a) = (Γ(a0) / Γ(a1)...Γ(ak)) * prod_k x_k ^ (a_k - 1)
    """

    def __init__(self, a: Optional[np.ndarray] = None):
        """
        Create a *Dirichlet* distribution.

        :param a: the alpha parameters
        """
        self.a = a
        super().__init__((gamma(np.sum(a)) * np.prod(symbols.x ** (self.a - 1))) / np.prod(gamma(a)))

    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        """
        pass

    def pdf(self, x: Union[np.ndarray, float]) -> Union[GenericDistribution, np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        if self.a is None:
            if isinstance(x, float):
                return GenericDistribution(self._formula.subs(symbols.x, x))
            else:
                raise ValueError(
                    "Dirichlet random variables should be of type float, but you gave " + str(type(x)) + ".\n"
                    "Since the parameters 'a' is undefined, the PDF is transformed into another generic\n"
                    "distribution over the undefined parameters after the random variable 'x' is fixed. Thus, if\n"
                    "an array of N random variables if given, N distributions should be generated,\n"
                    "which is currently not supported."
                )
        else:
            return gamma(np.sum(self.a)) * np.prod(x ** (self.a - 1)) / np.prod(gamma(self.a))

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.a is None:
            raise ValueError("The parameter 'a' is undefined.")
        return np.random.dirichlet(alpha=self.a, size=sample_size)
