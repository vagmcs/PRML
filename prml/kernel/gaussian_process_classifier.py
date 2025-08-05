# Dependencies
import numpy as np

# Project
from prml.kernel.kernel import Kernel
from prml.linear.classifier import Classifier


class GaussianProcessClassifier(Classifier):
    def __init__(self, kernel: Kernel, beta: float = 1e-4) -> None:
        super().__init__()
        self._kernel = kernel
        self._beta = beta
        self._x = None
        self._t = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        if x.ndim == 1:
            x = x[:, None]
        self._x = x
        self._t = t
        gram = self._kernel(x, x)
        self._precision = np.linalg.inv(gram + np.eye(len(gram)) * self._beta)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.ndim == 1:
            x = x[:, None]
        k = self._kernel(self._x, x)
        c = self._kernel(x, x)
        a_mu = k.T @ self._precision @ self._t
        a_sigma = c - np.sum(k.T @ self._precision * k.T, axis=-1)
        return 1 / (1 + np.exp(-a_mu)), a_sigma
