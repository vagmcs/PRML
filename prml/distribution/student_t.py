import numpy as np
import sympy as sym
import sympy.abc as symbols
from typing import Union
from scipy.special import gamma
from prml.distribution import GenericDistribution


class StudentT(GenericDistribution):
    """
    The Student's t-distribution:

    p(x|mu, lambda, nu) = ( Gamma((nu + 1) / 2) * sqrt(1 / (pi * nu)) (1 + (x**2 / nu)) ** (-(nu+1) / 2)) / Gamma(nu/2)
    """

    def __init__(self, nu: int):
        """
        Create a *Student t*-distribution.

        :param nu: the degrees of freedom
        """
        self.nu = nu
        super().__init__(
            ((gamma((nu + 1) / 2) * sym.sqrt(1 / (sym.pi * nu))) / gamma(nu / 2)) *
            (1 + (symbols.x**2 / nu)) ** (-(nu+1) / 2)
        )

    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        """
        # TODO: Requires Expectation-Maximization
        pass

    def pdf(self, x: Union[np.ndarray, float]) -> Union[GenericDistribution, np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        if self.nu is None:
            if isinstance(x, float):
                return GenericDistribution(self._formula.subs(symbols.x, x))
            else:
                raise ValueError(
                    "Student's t random variables should be of type float, but you gave " + str(type(x)) + ".\n"
                    "Since the parameter 'nu' is undefined, the PDF is transformed into another generic\n"
                    "distribution over the undefined parameters after the random variable 'x' is fixed. Thus, if\n"
                    "an array of N random variables if given, N distributions should be generated,\n"
                    "which is currently not supported."
                )
        else:
            return (
                ((gamma((self.nu + 1) / 2) * np.sqrt(1 / (np.pi * self.nu))) / gamma(self.nu / 2)) *
                (1 + (x**2 / self.nu)) ** (-(self.nu + 1) / 2)
            )

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.nu is None:
            raise ValueError("The parameter 'nu' is undefined.")
        return np.random.standard_t(df=self.nu, size=sample_size)
