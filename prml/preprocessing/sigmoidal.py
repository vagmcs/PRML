# Types
from typing import Union

# Dependencies
import numpy as np

# Project
from .basis_function import BasisFunction


class SigmoidFeature(BasisFunction):
    """
    Sigmoid basis functions.

    1 / (1 + exp(c @ (mean - x))
    """

    def __init__(self, mean: Union[int, float, np.ndarray], c: Union[int, float, np.ndarray] = 1):
        """
        Create sigmoid basis functions.

        :param mean: (D, 2) or (D, 1) array sigmoid function centers
        :param c : (D, 1) array, int or float coefficient
        """

        if isinstance(mean, int) or isinstance(mean, float):
            mean = np.array([[mean]])
        elif mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2, "Each mean should be vector not a matrix."

        if isinstance(c, int) or isinstance(c, float):
            if np.size(mean, 1) == 1:
                c = np.array([c])
            else:
                raise ValueError(f"Parameter c is a single value, while mean is of dimension {np.size(mean, 1)}.")
        else:
            assert c.ndim == 1, "Parameter c should be a vector."
            assert np.size(mean, 1) == len(c), "Mean and c should have the same dimension."

        self.mean = mean
        self.c = c

    def _sigmoid(self, x, mean):
        return np.tanh((x - mean) @ self.c * 0.5) * 0.5 + 0.5

    def transform(self, x: Union[int, float, np.ndarray]) -> np.ndarray:
        """
        Transform input array using sigmoid basis functions.

        :param x: (N, D) array of values, float or int
        :return: (N, D) array of sigmoid features
        """

        # Proper shape for 1-dimensional vectors
        if isinstance(x, np.ndarray):
            x = x[:, None] if x.ndim == 1 else x
        elif isinstance(x, int) or isinstance(x, float):
            x = np.array([[x]])
        else:
            raise ValueError(f'Incompatible type {type(x)}.')

        features = [np.ones(len(x))]  # create a list of ones for the bias parameter
        for m in self.mean:
            features.append(self._sigmoid(x, m))

        return np.asarray(features).T
