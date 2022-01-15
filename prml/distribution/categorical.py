# Types
from typing import Optional, Union

# Dependencies
import numpy as np
import sympy as sym

# Project
from .generic_distribution import GenericDistribution


class Categorical(GenericDistribution):
    """
    The **Categorical** distribution:

    p(x|mu) = prod_{k=1}^K mu_k^x_k
    """

    def __init__(self, mu: Optional[np.ndarray] = None, dim: Optional[int] = None):
        """
        Create a *Categorical* distribution.

        :param mu: the probability of the binary random variable to be True
        """
        if mu is None and dim is not None:
            self.D = dim
            mu = sym.MatrixSymbol("mu", self.D, 1)
            self.mu = None
        elif mu is not None:
            self.D = mu.shape[0]
            self.mu = mu
        else:
            raise AttributeError("Either provide the 'dim' argument or the parameters 'mu'.")

        x = sym.MatrixSymbol("x", self.D, 1)
        super().__init__(sym.prod(np.power(mu, x)))

    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        """
        # The maximum likelihood estimator for mu parameter in a Bernoulli
        # distribution is the sample mean.
        self.mu = np.mean(x, axis=0)
        # Update the formula to use the sample mean.
        self._formula = sym.prod(np.power(self.mu, sym.MatrixSymbol("x", self.D, 1)))

    def pdf(self, x: Union[np.ndarray]) -> Union[GenericDistribution, np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        # If mu is not defined then the PDF is transformed into another
        # generic distribution over the mu parameter.
        if self.mu is None:
            if x.shape[0] == self.D:
                return GenericDistribution(self._formula.subs(sym.MatrixSymbol("x", self.D, 1), sym.Matrix(x)))
            else:
                raise ValueError(
                    f"Categorical random variables should be one-hot vectors having dimensions (1, {str(self.D)})\n"
                    f"but you gave {str(x.shape)}. Since the parameters 'mu' are undefined, the PDF is\n"
                    "transformed into another generic distribution over the undefined parameters after the random\n"
                    f"variable 'x' is fixed. Thus, if a matrix of N random variables of dimension (1, {str(self.D)})\n"
                    "if given, N distributions should be generated, which is currently not supported."
                )
        else:
            return np.prod(self.mu ** x, axis=1)

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        if self.mu is None:
            raise ValueError("The parameters 'mu' are undefined.")

        sample = np.zeros((sample_size, self.D))
        sample[np.arange(sample_size), np.random.choice(np.arange(self.D), size=sample_size, p=self.mu)] = 1
        return sample
