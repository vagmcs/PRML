# Dependencies
import numpy as np
import sympy as sym
import sympy.abc as symbols

from .generic_distribution import GenericDistribution


class Bernoulli(GenericDistribution):
    """
    The **Bernoulli** distribution:

    p(x|mu) = mu^x * (1 - mu)^(x - 1)
    """

    def __init__(self, mu: sym.Symbol | int | float = symbols.mu):
        """
        Create a *Bernoulli* distribution.

        :param mu: the probability of the binary random variable to be true
        """
        self.mu = mu if isinstance(mu, (int, float)) else None
        super().__init__(mu**symbols.x * (1 - mu) ** (1 - symbols.x))

    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation on the parameters using the given data.

        :param x: an (N, D) array of data values
        """
        # The maximum likelihood estimator for mu parameter in a Bernoulli
        # distribution is the sample mean.
        self.mu = np.mean(x)
        # Update the formula to use the sample mean.
        self._formula = self.mu**symbols.x * (1 - self.mu) ** (1 - symbols.x)

    def pdf(self, x: np.ndarray | bool | int) -> GenericDistribution | np.ndarray | float:
        """
        Compute the probability density function (PDF) or the probability mass function
        (PMF) of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        # If mu is not defined then the PDF is transformed into another
        # generic distribution over the mu parameter.
        if self.mu is None:
            if isinstance(x, (bool, int)):
                return GenericDistribution(self._formula.subs(symbols.x, int(x)))
            else:
                raise ValueError(
                    "Bernoulli random variables should be of type bool or int, but you gave " + str(type(x)) + ".\n"
                    "Since the parameter 'mu' is undefined, the PDF is transformed into another generic distribution\n"
                    "over the 'mu' parameter after the random variable 'x' is fixed. Thus, if an array of N random\n"
                    "variables if given, N distributions should be generated, which is currently not supported."
                )

        else:
            return self.mu**x * (1 - self.mu) ** (1 - x)

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.mu is None:
            raise ValueError("The parameter 'mu' is undefined.")
        return (self.mu > np.random.uniform(size=sample_size)).astype(int)
