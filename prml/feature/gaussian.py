import numpy as np


class GaussianFeature(object):
    """
    Gaussian basis function

    exp(-0.5 * (x - mean) / var)
    """

    def __init__(self, mean: np.ndarray, var: np.ndarray):
        """
        Create Gaussian basis function

        :param mean: (n_features, ndim) or (n_features,) numpy array places to locate gaussian function at
        :param var: float variance of the gaussian function
        """
        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2
        assert isinstance(var, float) or isinstance(var, int)
        self.mean = mean
        self.var = var

    def _gauss(self, x, mean):
        return np.exp(-0.5 * np.sum(np.square(x - mean), axis=-1) / self.var)

    def transform(self, x):
        """
        Transform input array using gaussian features

        :param x: (sample_size, n) numpy array
        :return: (sample_size, n_features) numpy array of gaussian features
        """

        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert np.size(x, 1) == np.size(self.mean, 1)

        features = [np.ones(len(x))]
        for m in self.mean:
            features.append(self._gauss(x, m))
        return np.asarray(features).T
