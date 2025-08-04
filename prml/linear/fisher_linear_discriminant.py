# Types
from typing import Optional

# Dependencies
import numpy as np

# Project
from prml.distribution.gaussian import Gaussian

from .classifier import Classifier


class FisherLinearDiscriminant(Classifier):
    """
    Fisher's Linear discriminant classifier.
    """

    def __init__(self) -> None:
        """
        Creates a least squares classifier.
        """
        self._w: Optional[np.ndarray] = None
        self._threshold = 0.0

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Trains the classifier.

        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target classes
        """
        x_0 = x[t == 0]
        x_1 = x[t == 1]

        m_0 = np.mean(x_0, axis=0)
        m_1 = np.mean(x_1, axis=0)

        sw = np.cov(x_0, rowvar=False) + np.cov(x_1, rowvar=False)
        self._w = np.linalg.inv(sw) @ (m_1 - m_0)

        g0 = Gaussian()
        g0.ml(x_0 @ self._w)
        g1 = Gaussian()
        g1.ml(x_1 @ self._w)

        root = np.roots(
            [
                g1.var - g0.var,  # type: ignore
                2 * (g0.var * g1.mu - g1.var * g0.mu),  # type: ignore
                g1.var * g0.mu**2 - g0.var * g1.mu**2 - g1.var * g0.var * np.log(g1.var / g0.var),  # type: ignore
            ]
        )

        # keep the root that is located between of the two means
        self._threshold = root[0] if g0.mu < root[0] < g1.mu or g1.mu < root[0] < g0.mu else root[1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output :return (N,) array
            holding the predicted classes
        """
        return (x @ self._w > self._threshold).astype(np.int)  # type: ignore
