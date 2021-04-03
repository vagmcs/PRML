from typing import Optional, Union

import numpy as np
import sympy as sym
from generic_distribution import GenericDistribution
from scipy.special import gamma


class Dirichlet(GenericDistribution):
    """
    The **Dirichlet** distribution:

    p(x|a) = (Γ(a_0) / Γ(a_1)...Γ(a_K)) * prod_k x_k ^ (a_k - 1)
    """

    def __init__(self, alpha: Optional[np.ndarray] = None, dim: Optional[int] = None):
        """
        Create a *Dirichlet* distribution.

        :param alpha: the alpha parameters
        """
        if alpha is None and dim is not None:
            self.D = dim
            alpha = sym.MatrixSymbol('alpha', self.D, 1)
            self.alpha = None
        elif alpha is not None:
            self.D = alpha.shape[0]
            self.alpha = alpha
        else:
            raise AttributeError("Either provide the 'dim' argument or the parameters 'alpha'.")

        x = sym.MatrixSymbol('x', self.D, 1)
        super().__init__((gamma(np.sum(alpha)) * np.prod(x ** (alpha - 1))) / np.prod(gamma(alpha)))

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
        if self.alpha is None:
            if isinstance(x, float):
                return GenericDistribution(self._formula.subs(sym.MatrixSymbol('x', self.D, 1), sym.Matrix(x)))
            else:
                raise ValueError(
                    "Dirichlet random variables should be of type float, but you gave " + str(type(x)) + ".\n"
                    "Since the parameters 'alpha' is undefined, the PDF is transformed into another generic\n"
                    "distribution over the undefined parameters after the random variable 'x' is fixed. Thus, if\n"
                    "an array of N random variables if given, N distributions should be generated,\n"
                    "which is currently not supported."
                )
        else:
            return gamma(np.sum(self.alpha)) * np.prod(x ** (self.alpha - 1), axis=1) / np.prod(gamma(self.alpha))

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.alpha is None:
            raise ValueError("The parameter 'alpha' is undefined.")
        return np.random.dirichlet(alpha=self.alpha, size=sample_size)
