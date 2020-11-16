import math
from prml.distribution.distribution import Distribution


class Binomial(Distribution):

    def __init__(self, n: int, mu: float):
        self.n = n
        self.mu = mu

    def pdf(self, x: int):
        comb = math.factorial(self.n) / math.factorial(x) * math.factorial(self.n - x)
        return comb * (self.mu ** x) * (1 - self.mu) ** (self.n - x)
