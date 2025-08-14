# Standard Library
import abc

# Dependencies
import numpy as np
import sympy as sym
from numpy.typing import NDArray

# Project
from prml.helpers import array


class Distribution(abc.ABC):
    """
    Probability distribution base abstract class.
    """

    @property
    @abc.abstractmethod
    def formula(self) -> sym.Expr | None:
        """
        :return: the symbolic formula of the distribution
        """
        pass

    @property
    def to_latex(self) -> str:
        """
        :return: the latex representation of the distribution
        """
        return f"${sym.latex(self.formula)}$"

    @abc.abstractmethod
    def ml(self, x: NDArray[np.floating]) -> None:
        """
        Performs maximum likelihood estimation on the parameters using the given data.

        :param x: an (N, D) array of data values
        """
        pass

    def likelihood_iid(self, x: float | NDArray[np.floating]) -> float:
        """
        Compute the likelihood of the distribution on the given data, assuming that the
        data are independent and identically distributed.

        :param x: (N, D) array of data values or a single data value
        :return: the likelihood of the distribution
        """
        probs = array.to_array(self.pdf(x))
        return float(np.prod(probs))

    def log_likelihood_iid(self, x: float | NDArray[np.floating]) -> float:
        """
        Compute the logarithm of the likelihood of the distribution on the given data,
        assuming that the data are independent and identically distributed.

        :param x: (N, D) array of data values or a single data value
        :return: the logarithm of the likelihood of the distribution
        """
        probs = array.to_array(self.pdf(x))
        return float(np.sum(np.log(probs)))

    @abc.abstractmethod
    def pdf(self, x: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
        """
        Compute the probability density function (PDF) or the probability mass function
        (PMF) of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        pass

    @abc.abstractmethod
    def draw(self, sample_size: int) -> NDArray[np.floating]:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        pass

    @abc.abstractmethod
    def __mul__(self, other: "Distribution") -> "Distribution":
        """
        Symbolic multiplication of distributions.

        :param other: another distribution
        :return: a GenericDistribution object
        """

    def __str__(self) -> str:
        return str(self.formula)
