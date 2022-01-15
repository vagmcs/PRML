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

    def __init__(self, mean: Union[int, float, np.ndarray], sigma: Union[int, float] = 1):
        """
        Create sigmoid basis functions. Each basis function can either have a uni-variate
        or a multivariate mean and constant a constant coefficient. In the former case,
        the mean of each sigmoid is a single number, resulting in a 1-dimensional array,
        while the multivariate case, the mean of each sigmoid is an 1-dimensional array,
        resulting in a 2-dimensional array of means for all basis functions.

        :param mean: (D, 2) or (D, 1) array sigmoid function centers
        :param sigma: the spatial scale of the sigmoid basis functions
        """

        if isinstance(mean, (int, float)):
            self._mean = np.array([[mean]])
        elif isinstance(mean, np.ndarray) and mean.ndim <= 2:
            self._mean = mean[:, None] if mean.ndim == 1 else mean
        elif isinstance(mean, np.ndarray) and mean.ndim > 2:
            raise ValueError("Each mean should be an 1-dimensional array not a matrix (N-dimensional).")
        else:
            raise ValueError(f"Mean should be either an array or a number, but type '{type(mean)}' is given.")

        if not isinstance(sigma, (int, float)):
            raise ValueError(f"Spatial scale should be a number, but '{type(sigma)}' is given.")
        self._sigma = sigma

    def transform(self, x: Union[int, float, np.ndarray]) -> np.ndarray:
        """
        Transform input array using sigmoid basis functions.

        :param x: (N, D) array of values, float or int
        :return: (N, D) array of sigmoid features
        """

        # check if proper array is given or create one if not
        x = self._make_array(x)
        # create a list of ones for the bias parameter
        features = [np.ones(len(x))]

        for mean in self._mean:
            phi = 1 / (1 + np.exp(-np.sum(x - mean, axis=1) / self._sigma))
            features.append(phi)

        return np.asarray(features).T
