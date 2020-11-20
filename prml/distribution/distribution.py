import abc
import numpy as np
from typing import Union


class Distribution(metaclass=abc.ABCMeta):
    """
    Probability distribution base abstract class.
    """

    @property
    @abc.abstractmethod
    def formula(self):
        """
        :return: the symbolic formula of the distribution
        """
        pass

    @abc.abstractmethod
    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation on the parameters
        using the given data.

        :param x: an (N, D) array of data values
        """
        pass

    def likelihood_iid(self, x: Union[np.ndarray, float]) -> float:
        """
        Compute the likelihood of the distribution on the given data, assuming
        that the data are independent and identically distributed.

        :param x: (N, D) array of data values or a single data value
        :return: the likelihood of the distribution
        """
        return np.prod(self.pdf(x))

    def log_likelihood_iid(self, x: Union[np.ndarray, float]) -> float:
        """
        Compute the logarithm of the likelihood of the distribution on the given data,
        assuming that the data are independent and identically distributed.

        :param x: (N, D) array of data values or a single data value
        :return: the logarithm of the likelihood of the distribution
        """
        return sum(np.log(self.pdf(x)))

    @abc.abstractmethod
    def pdf(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        pass

    @abc.abstractmethod
    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        pass

    @abc.abstractmethod
    def __mul__(self, other: 'Distribution') -> 'Distribution':
        """
        Symbolic multiplication of distributions.

        :param other: another distribution
        :return: a GenericDistribution object
        """
        pass

    def __str__(self):
        return str(self.formula)
