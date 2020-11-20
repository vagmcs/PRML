import math
import numpy as np
import sympy as sym
import sympy.abc as symbols
from typing import Union
from prml.distribution import GenericDistribution


class Gaussian(GenericDistribution):
    """
    The **Gaussian** distribution:

    p(x|mu,var) = exp{-0.5 * (x - mu)^2 / var} / sqrt(2pi * var)
    """

    def __init__(self,
                 mu: Union[sym.Symbol, float] = symbols.mu,
                 var: Union[sym.Symbol, float] = sym.symbols('sigma^2')):
        """
        Create a *Gaussian* distribution.

        :param mu: the mean value
        :param var: the variance
        """
        self.mu = mu if isinstance(mu, float) else None
        self.var = var if isinstance(var, float) else None
        super().__init__(sym.exp(-(symbols.x - mu) ** 2 / (2 * var)) / sym.sqrt(2 * np.pi * var))

    def ml(self, x: np.ndarray, unbiased: bool = False) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        :param unbiased: if True it computes the unbiased variance (default is False)
        """
        # The maximum likelihood estimator for mu and sigma squared parameters in a Gaussian
        # distribution is the sample mean, and the sample variance (biased or unbiased).
        self.mu = np.mean(x)
        self.var = np.sum(np.power(x - self.mu, 2)) / (x.size - 1) if unbiased else np.var(x)
        # Update the formula to use the sample mean and variance.
        self._formula = sym.exp(-(symbols.x - self.mu) ** 2 / (2 * self.var)) / sym.sqrt(2 * np.pi * self.var)

    def pdf(self, x: Union[np.ndarray, float]) -> Union[GenericDistribution, np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        if self.mu is None or self.var is None:
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
            return np.exp(-(x - self.mu) ** 2 / (2 * self.var)) / np.sqrt(2 * np.pi * self.var)

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.mu is None or self.var is None:
            raise ValueError("The parameter 'mu' and/or 'var' is undefined.")
        return np.random.normal(loc=self.mu, scale=math.sqrt(self.var), size=sample_size)
