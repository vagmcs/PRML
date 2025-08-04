# Dependencies
import numpy as np

# Project
from prml.kernel.kernel import Kernel
from prml.linear.classifier import Classifier


class RelevanceVectorClassifier(Classifier):
    def __init__(self, kernel: Kernel) -> None:
        super().__init__()
        self._kernel = kernel
        self._alpha = None
        self._relevance_vectors = None
        self._relevance_labels = None

    @property
    def n_relevance_vectors(self) -> int:
        return 0 if self._relevance_vectors is None else len(self._relevance_vectors)

    @property
    def relevance_vectors(self) -> np.ndarray:
        return np.empty(0) if self._relevance_vectors is None else self._relevance_vectors

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x * 0.5) * 0.5 + 0.5

    def _map_estimation(self, x: np.ndarray, t: np.ndarray, w: np.ndarray):
        # IRLS
        for _ in range(100):
            y = self._sigmoid(x @ w)
            gradient = x.T @ (y - t) + self._alpha * w
            hessian = (x.T * y * (1 - y)) @ x + np.diag(self._alpha)
            w -= np.linalg.solve(hessian, gradient)
            return w, np.linalg.pinv(hessian)

    def fit(self, x: np.ndarray, t: np.ndarray, n_iter: int = 500) -> None:
        """
        Trains the classifier.

        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target classes
        """
        if x.ndim == 1:
            x = x[:, None]

        n = len(x)
        self._alpha = np.ones(n)
        self._beta = 1.0
        self._posterior_mean = np.zeros(n)

        # compute design or gram matrix
        k = x @ x.T if self._kernel is None else self._kernel(x, x)

        for _ in range(n_iter):
            prev_alpha = np.copy(self._alpha)
            self._posterior_mean, self._posterior_sigma = self._map_estimation(k, t, self._posterior_mean)

            # update alpha and beta
            gamma = 1 - self._alpha * np.diag(self._posterior_sigma)
            self._alpha = gamma / np.square(self._posterior_mean)
            np.clip(self._alpha, 0, 1e10, out=self._alpha)
            self._beta = (n - np.sum(gamma)) / np.sum((t - k @ self._posterior_mean) ** 2)

            # check for convergence
            if np.allclose(prev_alpha, self._alpha):
                break

        # store only relevance vectors
        relevance_vectors_idx = self._alpha < 1e8
        self._relevance_vectors = x[relevance_vectors_idx, :]
        self._relevance_labels = t[relevance_vectors_idx]
        self._alpha = self._alpha[relevance_vectors_idx]

        # update the gram matrix and the posterior parameters
        k = self._relevance_vectors @ self._relevance_vectors.T
        self._posterior_mean, self._posterior_sigma = self._map_estimation(
            k, self._relevance_labels, self._posterior_mean[relevance_vectors_idx]
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output :return (N,) array
            holding the predicted classes
        """
        if self._posterior_mean is None or self._posterior_sigma is None:
            raise ValueError("The model is not trained, thus predictions cannot be made!")

        if x.ndim == 1:
            x = x[:, None]

        # compute the predictive distribution
        k = x @ self._relevance_vectors.T if self._kernel is None else self._kernel(x, self._relevance_vectors)
        mean = k @ self._posterior_mean
        return (mean > 0).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._posterior_mean is None or self._posterior_sigma is None:
            raise ValueError("The model is not trained, thus predictions cannot be made!")

        if x.ndim == 1:
            x = x[:, None]

        k = x @ self._relevance_vectors.T if self._kernel is None else self._kernel(x, self._relevance_vectors)
        mean = k @ self._posterior_mean
        variance = np.sum(k @ self._posterior_sigma * k, axis=1)
        return self._sigmoid(mean / np.sqrt(1 + np.pi * variance / 8))
