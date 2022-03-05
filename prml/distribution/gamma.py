# Types
from typing import Union

# Dependencies
import numpy as np
import sympy as sym
import sympy.abc as symbols
from scipy.special import gamma

# Project
from .generic_distribution import GenericDistribution


class Gamma(GenericDistribution):
    """
    The Gamma distribution:

    p(x|a, b) = 1 / Gamma(a) * b^a * x^(a-1) * exp{-b * x}
    """

    def __init__(self, a: Union[sym.Symbol, int, float] = symbols.a, b: Union[sym.Symbol, int, float] = symbols.b):
        """
        Create a *Gamma* distribution.

        :param a: the mean value
        :param b: the variance
        """
        self.a = a if isinstance(a, (int, float)) else None
        self.b = b if isinstance(b, (int, float)) else None
        super().__init__((b**a * symbols.x ** (a - 1) * sym.exp(-b * symbols.x)) / gamma(a))

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
        if self.a is None or self.b is None:
            if isinstance(x, float):
                return GenericDistribution(self._formula.subs(symbols.x, x))
            else:
                raise ValueError(
                    "Gamma random variables should be of type float, but you gave " + str(type(x)) + ".\n"
                    "Since the parameters 'a' and/or 'b' is undefined, the PDF is transformed into another generic\n"
                    "distribution over the undefined parameters after the random variable 'x' is fixed. Thus, if\n"
                    "an array of N random variables if given, N distributions should be generated,\n"
                    "which is currently not supported."
                )
        else:
            return (self.b**self.a * x ** (self.a - 1) * np.exp(-self.b * x)) / gamma(self.a)

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.a is None or self.b is None:
            raise ValueError("The parameter 'a' and/or 'b' is undefined.")
        return np.random.gamma(shape=self.a, scale=1 / self.b, size=sample_size)
