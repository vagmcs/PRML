# Types
from typing import cast

# Dependencies
import numpy as np
from numpy.typing import NDArray

# Project
from prml.linear.classifier import Classifier


class BayesianLogisticRegression(Classifier):
    def __init__(self, alpha: float = 1) -> None:
        """
        :param alpha: precision parameter of the prior
        """
        self.alpha = alpha
        self.w_mean: np.ndarray | None = None
        self.w_precision: np.ndarray | None = None

    @staticmethod
    def _sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
        expr = 1.0 / (1.0 + np.exp(-x))
        return cast(NDArray[np.floating], expr)

    def fit(self, x: np.ndarray, t: np.ndarray, n_iter: int = 1000) -> None:
        w = np.zeros(np.size(x, 1))
        self.w_mean = np.copy(w)
        self.w_precision = self.alpha * np.eye(np.size(x, 1))

        for _ in range(n_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(x @ w)
            grad = x.T @ (y - t) + self.w_precision @ (w - self.w_mean)
            hessian = (x.T * y * (1 - y)) @ x + self.w_precision
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w_mean = w
        self.w_precision = hessian

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w_mean is None or self.w_precision is None:
            raise RuntimeError("Model mean and covariance are not initialized. Call 'fit' before 'predict'.")

        mu_a = x @ self.w_mean
        var_a = np.sum(np.linalg.solve(self.w_precision, x.T).T * x, axis=1)
        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
