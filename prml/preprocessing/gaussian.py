from typing import Union

import numpy as np
from basis_function import BasisFunction


class GaussianFeature(BasisFunction):
    """
    Gaussian basis functions.

    exp(-0.5 * (x - mean) / var)
    """

    def __init__(self, mean: Union[int, float, np.ndarray], var: Union[int, float]):
        """
        Create Gaussian basis functions.

        :param mean: (M, 2) or (M, 1) array of Gaussian function locations (mean)
        :param var: variance (spatial scale) of the gaussian basis functions
        """

        assert isinstance(mean, int) or isinstance(mean, float) or isinstance(mean, np.ndarray), \
            f"mean should be of type 'Union[int, float, array]', but type '{type(mean)}' was found."

        if isinstance(mean, int) or isinstance(mean, float):
            mean = np.array([[mean]])
        elif mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2, "Each mean should be vector not a matrix."

        assert isinstance(var, float) or isinstance(var, int), "Variance (spatial scale) should be 'float' or 'int'."

        self.mean = mean
        self.var = var

    def _gauss(self, x, mean):
        return np.exp(-0.5 * np.sum(np.square(x - mean), axis=-1) / self.var)

    def transform(self, x: Union[int, float, np.ndarray]) -> np.ndarray:
        """
        Transform input array using gaussian basis functions.

        :param x: (N, D) array of values or a single float or int value
        :return: (N, D) array of gaussian features
        """

        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x[:, None]
            else:
                assert x.ndim == 2, "Input data should be an (N, D) array, where N is the number of samples" \
                                    " and D is the dimension of each sample."
        elif isinstance(x, int) or isinstance(x, float):
            x = np.array([[x]])
        else:
            raise ValueError(f'Incompatible type {type(x)}.')

        assert np.size(x, 1) == np.size(self.mean, 1), \
            "Input data should have the same dimension as the mean of the Gaussian basis function."

        features = [np.ones(len(x))]  # create a list of ones for the bias parameter
        for m in self.mean:
            features.append(self._gauss(x, m))

        return np.asarray(features).T
