import numpy as np
from typing import Union


class SigmoidFeature(object):
    """
    Sigmoid features:

    1 / (1 + exp(c @ (mean - x))
    """

    def __init__(self, mean: np.ndarray, c: Union[np.ndarray, float, int] = 1):
        """
        Create sigmoid features

        :param mean: (N, D) or (n_features,) numpy array center of sigmoid function
        :param c : (D,) array, int or float coefficient
        """

        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2

        if isinstance(c, int) or isinstance(c, float):
            if np.size(mean, 1) == 1:
                c = np.array([c])
            else:
                raise ValueError("mismatch of dimension")
        else:
            assert c.ndim == 1
            assert np.size(mean, 1) == len(c)

        self.mean = mean
        self.c = c

    def _sigmoid(self, x, mean):
        return np.tanh((x - mean) @ self.c * 0.5) * 0.5 + 0.5

    def transform(self, x):
        """
        Transform input array using sigmoid features

        :param x: (sample_size, n) numpy array
        :return: (sample_size, n_features) numpy array of sigmoid features
        """

        # Proper shape for 1-dimensional vectors
        x = x[:, None] if x.ndim == 1 else x

        features = [np.ones(len(x))]
        for m in self.mean:
            features.append(self._sigmoid(x, m))
        return np.asarray(features).T
