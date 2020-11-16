import numpy as np
from typing import Optional
from prml.distribution.distribution import Distribution


class MultivariateGaussian(Distribution):
    """
    The multivariate Gaussian distribution

    p(x|mu, cov) = exp{-0.5 * (x - mu)^T @ cov^-1 @ (x - mu)} / (2pi)^(D/2) / |cov|^0.5
    """

    def __init__(self, mu: Optional[np.ndarray], covariance: Optional[np.ndarray]):
        self.mu = mu
        self.covariance = covariance
        self.D = mu.size

    def pdf(self, x: np.ndarray):
        d = x - self.mu
        return (
            1.0 / (np.sqrt((2 * np.pi) ** self.D * np.linalg.det(self.covariance))) *
            np.exp(-(np.linalg.solve(self.covariance, d).T.dot(d)) / 2)
        )
