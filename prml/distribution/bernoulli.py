from prml.distribution.distribution import Distribution


class Bernoulli(Distribution):

    def __init__(self, mu: float):
        self.mu = mu

    def pdf(self, x: int):
        return self.mu ** x * (1 - self.mu) ** (1 - x)
