# Types
from typing import Tuple, Union

# Dependencies
import numpy as np

# Project
from prml.kernel.kernel import Kernel
from prml.linear.regression import Regression


class GaussianProcessRegression(Regression):
    """
    Gaussian process regression model

    p(t_n+1|t_n) = N(t_n+1|k^T C_N^-1 t_N, c - k^T C_N^-1 k)
    """

    def __init__(self, kernel: Kernel, beta: Union[int, float]) -> None:
        """
        Creates a linear regression model.
        """
        super().__init__()
        self._kernel = kernel
        self._beta = beta
        self._inv_c = None
        self._x = None
        self._t = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        n = x.shape[0]
        self._inv_c = np.linalg.pinv(self._kernel(x, x) + np.eye(n) * (1 / self._beta))
        self._x = x.copy()
        self._t = t.copy()

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = self._kernel(self._x, x)
        c = self._kernel(x, x, pairwise=False)
        k_inv_c = k.T @ self._inv_c
        mu = k_inv_c @ self._t
        sigma = c - np.sum(k_inv_c * k.T, axis=-1)
        return mu, sigma
