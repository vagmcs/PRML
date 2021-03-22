import numpy as np
from typing import Union
from prml.linear import BayesianRegression


class EvidenceApproximation(BayesianRegression):
    """

    """

    def __init__(self, alpha: Union[int, float] = 1, beta: Union[int, float] = 1):
        """

        """
        super().__init__(alpha, beta)

    def fit(self, x: np.ndarray, t: np.ndarray, n_iter: int = 100) -> None:
        """
        Maximizes the evidence function over the hyper-parameters alpha and beta
        given a training dataset.

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        :param n_iter: number of iterations
        """
        x_product = x.T @ x
        eigenvalues = np.linalg.eigvalsh(x_product)
        n = len(t)
        for _ in range(n_iter):
            prev_alpha = self.alpha
            prev_beta = self.beta

            super().fit(x, t)  # estimate mean and precision
            gamma = np.sum(eigenvalues / (self.alpha + eigenvalues))

            self.alpha = gamma / (self.mean.T @ self.mean)
            self.beta = (n - gamma) / np.sum(np.square(t - x @ self.mean))

            # check for convergence
            if np.allclose([prev_alpha, prev_beta], [self.alpha, self.beta]):
                break

    def _log_posterior(self, x: np.ndarray, t: np.ndarray, w: np.ndarray):
        log_prior = -0.5 * self.alpha * np.sum(w ** 2)
        log_likelihood = -0.5 * self.beta * np.square(t - x @ w).sum()
        return log_likelihood + log_prior

    def log_evidence(self, x: np.ndarray, t: np.ndarray):
        """
        Logarithm of the evidence function.

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        :return: log evidence
        """

        n = len(t)
        d = np.size(x, 1)
        return 0.5 * (
                d * np.log(self.alpha)
                + n * np.log(self.beta)
                - np.linalg.slogdet(self.precision)[1]
                - n * np.log(2 * np.pi)
        ) + self._log_posterior(x, t, self.mean)
