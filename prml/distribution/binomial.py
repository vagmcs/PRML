from typing import Union

import numpy as np
import sympy as sym
import sympy.abc as symbols
from generic_distribution import GenericDistribution
from scipy.special import factorial


class Binomial(GenericDistribution):
    """
    The **Binomial** distribution:

    p(x|N, mu) = (N!/(N - x)!x!) * mu^x * (1 - mu)^(N - x)
    """

    def __init__(self, n: int, mu: Union[sym.Symbol, int, float] = symbols.mu):
        """
        Creates a *Binomial* distribution.

        :param n: number of trials
        :param mu: the probability of the binary random variable to be true
        """
        self.n = n
        self.mu = mu if isinstance(mu, (int, float)) else None
        super().__init__(sym.binomial(n, symbols.x) * mu ** symbols.x * (1 - mu) ** (n - symbols.x))

    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        """
        # The maximum likelihood estimator for mu parameter in a Binomial
        # distribution is the sample mean, the same as in the Bernoulli distribution.
        self.mu = np.mean(x)
        # Update the formula to use the sample mean.
        self._formula = sym.binomial(self.n, symbols.x) * self.mu ** symbols.x * (1 - self.mu) ** (self.n - symbols.x)

    def pdf(self, x: Union[np.ndarray, int]) -> Union[GenericDistribution, np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        if self.mu is None:
            if isinstance(x, (bool, int)):
                return GenericDistribution(self._formula.subs(symbols.x, int(x)))
            else:
                raise ValueError(
                    "Binomial random variables should be of type bool or int, but you gave " + str(type(x)) + ".\n"
                    "Since the parameter 'mu' is undefined, the PDF is transformed into another generic distribution\n"
                    "over the 'mu' parameter after the random variable 'x' is fixed. Thus, if an array of N random\n"
                    "variables if given, N distributions should be generated, which is currently not supported."
                )
        else:
            binomial_coefficient = factorial(self.n) / (factorial(x) * factorial(self.n - x))
            return binomial_coefficient * (self.mu ** x) * (1 - self.mu) ** (self.n - x)

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.mu is None:
            raise ValueError("The parameter 'mu' is undefined.")
        return np.random.binomial(n=self.n, p=self.mu, size=sample_size)
