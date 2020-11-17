import math
import numpy as np
from prml.distribution.distribution import Distribution


class Binomial(Distribution):
    """

    """

    def __init__(self, n: int, mu: float):
        self.n = n
        self.mu = mu

    def pdf(self, x: int) -> float:
        """

        :param x:
        :return:
        """
        comb = math.factorial(self.n) / (math.factorial(x) * math.factorial(self.n - x))
        return comb * (self.mu ** x) * (1 - self.mu) ** (self.n - x)

    def draw(self, sample_size: int) -> np.ndarray:
        """

        :param sample_size:
        :return:
        """
        return np.random.binomial(self.n, self.mu, (sample_size,))
