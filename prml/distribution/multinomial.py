# Types
from typing import Optional, Union

# Dependencies
import numpy as np
import sympy as sym
from scipy.special import factorial

# Project
from .generic_distribution import GenericDistribution


class Multinomial(GenericDistribution):
    """
    The **Multinomial** distribution:

    p(x|N,mu) = (N! / x_1!x_2!...x_K!) prod_{k=1}^K mu_k^x_k
    """

    def __init__(self, n: int, mu: Optional[np.ndarray] = None, dim: Optional[int] = None):
        """
        Create a *Multinomial* distribution.

        :param n: number of trials
        :param mu: the probability of the binary random variable to be True
        :param dim: the dimension of the multinomial random variables
        """
        self.n = n
        if mu is None and dim is not None:
            self.D = dim
            mu = sym.MatrixSymbol('mu', self.D, 1)
            self.mu = None
        elif mu is not None:
            self.D = mu.shape[0]
            self.mu = mu
        else:
            raise AttributeError("Either provide the 'dim' argument or the parameters 'mu'.")

        x = sym.MatrixSymbol('x', self.D, 1)
        super().__init__(sym.binomial(n, sym.prod(x)) * sym.prod(np.power(mu, x)[0]))

    def ml(self, x: np.ndarray) -> None:
        pass

    def pdf(self, x: Union[np.ndarray]) -> Union[GenericDistribution, np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        if self.mu is None:
            if x.shape[0] == self.D:
                return GenericDistribution(self._formula.subs(sym.MatrixSymbol('x', self.D, 1), sym.Matrix(x)))
            else:
                raise ValueError(
                    "Multinomial random variables should be one-hot vectors having dimensions (1, " + str(self.D) +
                    "),\nbut you gave " + str(x.shape) + ". Since the parameters 'mu' are undefined, the PDF is\n"
                    "transformed into another generic distribution over the undefined parameters after the random\n"
                    "variable 'x' is fixed. Thus, if a matrix of N random variables of dimension (1, " + str(self.D) +
                    ") if given, N distributions should be generated, which is currently not supported."
                )
        else:
            multinomial_coefficient = factorial(self.n) / np.product(factorial(x))
            return multinomial_coefficient * np.prod(self.mu ** x, axis=0)

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.mu is None:
            raise ValueError("The parameter 'mu' is undefined.")
        return np.random.multinomial(n=self.n, pvals=self.mu, size=sample_size)
