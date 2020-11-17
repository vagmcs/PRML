import numpy as np
from typing import Optional
from prml.distribution.distribution import Distribution


class Bernoulli(Distribution):
    """
    Bernoulli distribution.

    p(x|mu) = mu^x * (1 - mu)^(x - 1)
    """

    def __init__(self, mu: Optional[float] = None):
        """
        :param mu: the probability of the binary random variable to be True
        """
        self.mu = mu
        if mu:
            assert 0 <= mu <= 1, "The mu parameter must be in [0, 1]."

    def ml(self, x: np.ndarray) -> None:
        """
        Performs maximum likelihood estimation for the mu parameter.

        :param x: an array of observations
        """
        self.mu = np.mean(x)

    def pdf(self, x: int) -> float:
        """
        Compute the PDF of the given value.

        :param x: a value for the random variable
        :return: the PDF value
        """
        return self.mu ** x * (1 - self.mu) ** (1 - x)

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw a sample from the distribution.

        :param sample_size: the number of samples to draw
        :return: an array of samples
        """
        return (self.mu > np.random.uniform(size=(sample_size,))).astype(int)
