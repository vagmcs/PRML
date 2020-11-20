import numpy as np
import sympy as sym
from typing import Optional, Union
from prml.distribution import GenericDistribution


class MultivariateGaussian(GenericDistribution):
    """
    The Multivariate **Gaussian** distribution:

    p(x|mu, cov) = exp{-0.5 * (x - mu)^T @ cov^-1 @ (x - mu)} / (2pi)^(D/2) / |cov|^0.5
    """

    def __init__(self, mu: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None, dim: Optional[int] = None):
        """
         Create a multivariate *Gaussian* distribution.

        :param mu: the mean column vector
        :param cov: the covariance matrix
        :param dim: the dimension of the Gaussian random variables
        """
        if mu is None and cov is None and dim is not None:
            self.D = dim
            self.mu = None
            mu = sym.MatrixSymbol('mu', self.D, 1)
            self.cov = None
            cov = sym.MatrixSymbol('Sigma', self.D, self.D)
        elif not dim and (mu is not None or cov is not None):
            self.D = mu.shape[0] if mu is not None else cov.shape[0]
            self.mu = mu
            mu = mu if mu is not None else sym.MatrixSymbol('mu', self.D, 1)
            self.cov = cov
            cov = sym.Matrix(cov) if cov is not None else sym.MatrixSymbol('Sigma', self.D, self.D)
        elif mu is not None and cov is not None and dim is not None:
            self.D = dim
            self.mu = mu
            self.cov = cov
        else:
            raise AttributeError("Either provide the 'dim' argument or one of the parameters ('mu' or 'cov').")

        x = sym.MatrixSymbol('x', self.D, 1)
        super().__init__(
            1 / (sym.sqrt((2 * np.pi) ** 2 * sym.det(cov))) *
            sym.exp(-0.5 * ((x - mu).T * cov.inv() * (x - mu)))
        )

    def ml(self, x: np.ndarray) -> None:
        pass

    def pdf(self, x: np.ndarray) -> Union[GenericDistribution, np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        if self.mu is None or self.cov is None:
            if x.shape[0] == self.D:
                return GenericDistribution(self._formula.subs(sym.MatrixSymbol('x', self.D, 1), sym.Matrix(x)))
            else:
                raise ValueError(
                    "Multivariate Gaussian random variables should have dimensions (1, " + str(self.D) + "),\n"
                    "but you gave " + str(x.shape) + ". Since the parameters 'mu' or 'cov' are undefined, the PDF is\n"
                    "transformed into another generic distribution over the undefined parameters after the random\n"
                    "variable 'x' is fixed. Thus, if a matrix of N random variables of dimension (1, " + str(self.D) +
                    ") if given, N distributions should be generated, which is currently not supported."
                )
        else:
            d = x - self.mu
            return (
                1 / (np.sqrt((2 * np.pi) ** self.D * np.linalg.det(self.cov))) *
                np.exp(-0.5 * (np.linalg.solve(self.cov, d).T.dot(d)))
            )

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.mu is None or self.cov is None:
            raise ValueError("The parameter 'mu' and/or 'cov' is undefined.")
        return np.random.multivariate_normal(mean=self.mu, cov=self.cov, size=sample_size)
