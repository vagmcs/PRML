# Dependencies
import numpy as np
from numpy.typing import NDArray

# Project
from prml.helpers import array
from prml.helpers.array import Axis


class OneHotEncoder:
    """
    OneHotEncoder encodes arrays of classes into 1-of-K class vectors.
    """

    def __init__(self, k: int | None = None) -> None:
        """
        Creates an encoder given a number of classes. In case the number classes is not
        provided, they are automatically inferred during the encoding step from the
        given array.

        :param k: the number of classes, defaults to None
        """
        self._k = k

    def encode(self, class_indices: NDArray[np.uint32]) -> NDArray[np.uint8]:
        """
        Encodes an array of class indices into one-of-K coding.

        :param class_indices: (N,) array of non-negative class indices
        :return: (N, K) array holding the one-of-K encodings
        """

        n_classes = int(np.max(class_indices)) + 1 if self._k is None else self._k
        return np.eye(n_classes, dtype=np.uint8)[class_indices]

    @staticmethod
    def decode(onehot: NDArray[np.uint8]) -> NDArray[np.uint32]:
        """
        Decodes the one-of-K code into class indices.

        :param onehot: (N, K) array containing one-of-K codings
        :return: (N,) array of class indices
        """
        return array.cast_uint(np.argmax(onehot, axis=Axis.COLS))
