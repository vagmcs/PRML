# Dependencies
import numpy as np

from .basis_function import BasisFunction


class GaussianFeature(BasisFunction):
    """
    Gaussian basis functions.

    exp(-0.5 * ||x - mean||^2 / sigma)
    """

    def __init__(self, mean: int | float | np.ndarray, sigma: int | float):
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

    def transform(self, x: int | float | np.ndarray) -> np.ndarray:
        """
        Transform input array using gaussian basis functions.

        :param x: (N, D) array of values, float or int
        :return: (N, D) array of gaussian features
        """

        # check if proper array is given or create one if not
        x = self._make_array(x)

        # check if the given input array has the same dimension as the mean of the Gaussian
        if np.size(x, 1) != np.size(self._mean, 1):
            raise ValueError(
                "Input data instances must have the same dimension as the mean of each Gaussian basis function."
            )

        # create a list of ones for the bias parameter
        features = [np.ones(len(x))]

        for mean in self._mean:
            # in the general case the numerator equals to ||x - mu||^2
            phi = np.exp(-0.5 * np.linalg.norm(x - mean, axis=1) ** 2 / self._sigma)
            features.append(phi)

        return np.asarray(features).T
