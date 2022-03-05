# Types
from typing import Union

# Dependencies
import numpy as np

# Project
from .bayesian_regression import BayesianRegression


class EvidenceApproximation(BayesianRegression):
    """
    Sets the (hyper) parameters alpha and beta to specific values and approximates
    them by maximizing the marginal likelihood or evidence function obtained by
    integrating over the model parameters. This framework is known in statistics as
    empirical bayes, type 2 maximum likelihood, generalized maximum likelihood or
    evidence approximation.
    """

    def __init__(self, alpha: Union[int, float] = 1, beta: Union[int, float] = 1):
        super().__init__(alpha, beta)

    def fit(self, x: np.ndarray, t: np.ndarray, n_iter: int = 100) -> None:
        """
        Maximizes the evidence function over the (hyper) parameters alpha and beta
        given a training dataset.

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        :param n_iter: number of iterations
        """
        x_product = x.T @ x
        eigenvalues = np.linalg.eigvalsh(x_product)
        n = len(t)
        for _ in range(n_iter):
            prev_alpha = self._alpha
            prev_beta = self._beta

            super().fit(x, t)  # estimate mean and precision
            gamma = np.sum(eigenvalues / (self._alpha + eigenvalues))

            self._alpha = gamma / (self._mean.T @ self._mean)  # type: ignore
            self._beta = (n - gamma) / np.sum(np.square(t - x @ self._mean))

            # check for convergence
            if np.allclose([prev_alpha, prev_beta], [self._alpha, self._beta]):
                break

    def _log_posterior(self, x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
        log_prior = -0.5 * self._alpha * np.sum(w**2)
        log_likelihood = -0.5 * self._beta * np.square(t - x @ w).sum()
        return log_likelihood + log_prior  # type: ignore

    def log_evidence(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Logarithm of the evidence function.

        :param x: (N, D) numpy array holding the input training data
        :param t: (N,) numpy array holding the target values
        :return: log evidence
        """

        n = len(t)
        d = np.size(x, 1)
        return 0.5 * (  # type: ignore
            d * np.log(self._alpha)
            + n * np.log(self._beta)
            - np.linalg.slogdet(self._precision)[1]  # type: ignore
            - n * np.log(2 * np.pi)
        ) + self._log_posterior(
            x, t, self._mean  # type: ignore
        )
