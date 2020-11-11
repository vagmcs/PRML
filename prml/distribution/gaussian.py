import numpy as np
from typing import Optional
from prml.distribution.distribution import Distribution


class Gaussian(Distribution):
    """
    """

    def __init__(self, mu: Optional[float], var: Optional[float]):
        self.mu = mu
        self.var = var

    def pdf(self, x):
        d = x - self.mu
        return np.exp(-d ** 2 / (2 * self.var)) / np.sqrt(2 * np.pi * self.var)
