# Dependencies
import numpy as np

# Project
from prml.kernel.kernel import Kernel
from prml.linear.regression import Regression


class GaussianProcessRegression(Regression):
    """
    Gaussian process regression model.

    p(t_n+1|t_n) = N(t_n+1|k^T C_N^-1 t_N, c - k^T C_N^-1 k)
    """

    def __init__(self, kernel: Kernel, beta: int | float) -> None:
        """
        Creates a linear regression model.
        """
        super().__init__()
        self._kernel = kernel
        self._beta = beta
        self._precision = None
        self._x = None
        self._t = None

    def fit(self, x: np.ndarray, t: np.ndarray, iterations: int = 0, learning_rate: float = 0.001) -> None:
        n = x.shape[0]
        self._x = x.copy()
        self._t = t.copy()

        identity = np.eye(n)
        k = self._kernel(x, x)
        self._precision = np.linalg.inv(k + identity / self._beta)

        for i in range(iterations):
            gradient = self._kernel.derivative(x, x)
            update = -np.trace(self._precision @ gradient.T) + t @ self._precision @ gradient @ self._precision @ t

            self._kernel.theta += learning_rate * update
            k = self._kernel(x, x)
            self._precision = np.linalg.inv(k + identity / self._beta)

            if i % 100 == 0:
                ll_error = 0.5 * (
                    np.linalg.slogdet(k + identity / self._beta)[1]
                    + t @ self._precision @ t
                    + len(t) * np.log(2 * np.pi)
                )
                print(f"-- Iterations {i}: {ll_error}")

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = self._kernel(self._x, x)
        c = self._kernel(x, x, pairwise=False)
        k_precision = k.T @ self._precision
        mu = k_precision @ self._t
        sigma = c - np.sum(k_precision * k.T, axis=-1)
        return mu, sigma
