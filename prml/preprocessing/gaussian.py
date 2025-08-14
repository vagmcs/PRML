# Dependencies
import numpy as np
from numpy.typing import NDArray

# Project
from prml.helpers import array
from prml.helpers.array import Axis
from prml.preprocessing.basis_function import BasisFunction


class GaussianFeature(BasisFunction):
    """
    Gaussian basis functions.

    exp(-0.5 * ||x - mean||^2 / sigma)
    """

    def __init__(self, mean: int | float | NDArray[np.floating], sigma: int | float):
        """
        Create Gaussian basis functions. Each basis function can either be a uni-variate
        or a multivariate Gaussian having constant across dimensions. In the former
        case, the mean of each Gaussian is a single number, resulting in a 1-dimensional
        array, while the multivariate case, the mean of each Gaussian is an
        1-dimensional array, resulting in a 2-dimensional array of means for all basis
        functions.

        Note that 1 / sigma is sometimes also called gamma.

        :param mean: array of Gaussian function locations (mean)
        :param sigma: the spatial scale of the gaussian basis functions
        """

        # check mean
        if isinstance(mean, (int, float, np.ndarray)):
            self._mean = array.to_array(mean)
        else:
            raise ValueError(f"Mean should be either an array or a number, but type '{type(mean)}' is given.")

        # check variance
        if not isinstance(sigma, (int, float)):
            raise ValueError(f"Spatial scale should be a number, but '{type(sigma)}' is given.")
        self._sigma = sigma

    def transform(self, x: float | np.ndarray) -> np.ndarray:
        """
        Transform input array using gaussian basis functions.

        :param x: (N, D) array of values, float or int
        :return: (N, D) array of gaussian features
        """

        # check if proper array is given or create one if not
        x = array.to_array(x)

        # check if the given input array has the same dimension as the mean of the Gaussian
        if np.size(x, Axis.COLS) != np.size(self._mean, Axis.COLS):
            raise ValueError(
                "Input data instances must have the same dimension as the mean of each Gaussian basis function."
            )

        # create a list of ones for the bias parameter
        features = [np.ones(len(x))]

        for mean in self._mean:
            # in the general case the numerator equals to ||x - mu||^2
            phi = np.exp(-0.5 * np.linalg.norm(x - mean, axis=Axis.COLS) ** 2 / self._sigma)
            features.append(phi)

        return np.asarray(features).T
