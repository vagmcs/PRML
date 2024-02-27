# Dependencies
import numpy as np

# Project
from prml.kernel.kernel import Kernel
from prml.linear.regression import Regression


class RelevanceVectorRegressor(Regression):
    def __init__(self, kernel: Kernel) -> None:
        super().__init__()
        self._kernel = kernel
        self._alpha = None
        self._beta = None
        self._posterior_mean = None
        self._posterior_sigma = None
        self._relevance_vectors = None
        self._relevance_labels = None

    @property
    def n_relevance_vectors(self) -> int:
        return 0 if self._relevance_vectors is None else len(self._relevance_vectors)

    @property
    def relevance_vectors(self) -> np.ndarray:
        return np.empty(0) if self._relevance_vectors is None else self._relevance_vectors

    @property
    def relevance_labels(self) -> np.ndarray:
        return np.empty(0) if self._relevance_labels is None else self._relevance_labels

    def fit(self, x: np.ndarray, t: np.ndarray, n_iter: int = 1000) -> None:
        if x.ndim == 1:
            x = x[:, None]

        n = len(x)
        self._alpha = np.ones(n)
        self._beta = 1.0

        # compute design or gram matrix
        k = x @ x.T if self._kernel is None else self._kernel(x, x)

        for _ in range(n_iter):
            prev_params = np.hstack([self._alpha, self._beta])

            self._posterior_sigma = np.linalg.inv(np.diag(self._alpha) + self._beta * k.T @ k)
            self._posterior_mean = self._beta * self._posterior_sigma @ k.T @ t

            # update alpha and beta
            gamma = 1 - self._alpha * np.diag(self._posterior_sigma)
            self._alpha = gamma / np.square(self._posterior_mean)
            np.clip(self._alpha, 0, 1e10, out=self._alpha)
            self._beta = (n - np.sum(gamma)) / np.sum((t - k @ self._posterior_mean) ** 2)

            # check for convergence
            if np.allclose(prev_params, np.hstack([self._alpha, self._beta])):
                break

        # store only relevance vectors
        relevance_vectors_idx = self._alpha < 1e8
        self._relevance_vectors = x[relevance_vectors_idx, :]
        self._relevance_labels = t[relevance_vectors_idx]
        self._alpha = self._alpha[relevance_vectors_idx]

        # update the gram matrix and the posterior parameters
        k = self._relevance_vectors @ self._relevance_vectors.T
        self._posterior_sigma = np.linalg.inv(np.diag(self._alpha) + self._beta * k.T @ k)
        self._posterior_mean = self._beta * self._posterior_sigma @ k.T @ self._relevance_labels

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._posterior_mean is None or self._posterior_sigma is None:
            raise ValueError("The model is not trained, thus predictions cannot be made!")

        if x.ndim == 1:
            x = x[:, None]

        # compute the predictive distribution
        k = x @ self._relevance_vectors.T if self._kernel is None else self._kernel(x, self._relevance_vectors)
        mean = k @ self._posterior_mean
        variance = np.sqrt(1 / self._beta + np.sum(k @ self._posterior_sigma * k, axis=1))
        return mean, variance
