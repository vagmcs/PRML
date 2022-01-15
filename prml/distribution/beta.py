# Types
from typing import Union

# Dependencies
import numpy as np
import sympy as sym
import sympy.abc as symbols
from scipy.special import gamma

# Project
from .generic_distribution import GenericDistribution


class Beta(GenericDistribution):
    """
    The **Beta** distribution:

    p(x|a, b) = (Γ(a + b) / Γ(a)Γ(b)) * x^(a - 1) * (1 - x)^(b - 1)
    """

    def __init__(self, a: Union[sym.Symbol, int, float] = symbols.a, b: Union[sym.Symbol, int, float] = symbols.b):
        """
        Create a *Beta* distribution.

        :param a: number of ones
        :param b: number of zeros
        """
        self.a = a if isinstance(a, (int, float)) else None
        self.b = b if isinstance(b, (int, float)) else None
        super().__init__(
            sym.gamma(a + b) / (sym.gamma(a) * sym.gamma(b)) * symbols.x ** (a - 1) * (1 - symbols.x) ** (b - 1)
        )

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
                    "Beta random variables should be of type float, in general, but you gave " + str(type(x)) + ".\n"
                    "Since the parameters 'a' or 'b' are undefined, the PDF is transformed into another generic\n"
                    "distribution over the undefined parameters after the random variable 'x' is fixed. Thus, if\n"
                    "an array of N random variables if given, N distributions should be generated,\n"
                    "which is currently not supported."
                )
        else:
            return (
                gamma(self.a + self.b)
                / (gamma(self.a) * gamma(self.b))
                * np.power(x, self.a - 1)
                * np.power(1 - x, self.b - 1)
            )

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.a is None or self.b is None:
            raise ValueError("The parameter 'a' and/or 'b' is undefined.")
        return np.random.beta(a=self.a, b=self.b, size=sample_size)
