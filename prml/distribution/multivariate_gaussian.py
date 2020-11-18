import numpy as np
from typing import Optional, Union
from prml.distribution import Distribution


class MultivariateGaussian(Distribution):
    """
    Multivariate **Gaussian** distribution:

    p(x|mu, cov) = exp{-0.5 * (x - mu)^T @ cov^-1 @ (x - mu)} / (2pi)^(D/2) / |cov|^0.5
    """

    def __init__(self, mu: Optional[np.ndarray], covariance: Optional[np.ndarray]):
        """

        :param mu:
        :param covariance:
        """
        self.mu = mu
        self.covariance = covariance
        self.D = mu.size

    def ml(self, x: np.ndarray) -> None:
        pass

    def pdf(self, x: np.ndarray) -> Union[np.ndarray, float]:
        """
        Compute the probability density function (PDF) or the probability mass function (PMF)
        of the given values for the random variables.

        :param x: (N, D) array of values or a single value for the random variables
        :return: the probability density function value
        """
        d = x - self.mu
        return (
            1.0 / (np.sqrt((2 * np.pi) ** self.D * np.linalg.det(self.covariance))) *
            np.exp(-(np.linalg.solve(self.covariance, d).T.dot(d)) / 2)
        )

    def draw(self, sample_size: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        return np.random.multivariate_normal(mean=self.mu, cov=self.covariance, size=sample_size)
