# Dependencies
import numpy as np


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

    def encode(self, class_indices: np.ndarray) -> np.ndarray:
        """
        Encodes an array of class indices into one-of-K coding.

        :param class_indices: (N,) array of non-negative class indices
        :return: (N, K) array holding the one-of-K encodings
        """

        n_classes = np.max(class_indices) + 1 if self._k is None else self._k
        return np.eye(n_classes)[class_indices]  # type: ignore

    @staticmethod
    def decode(onehot: np.ndarray) -> np.ndarray:
        """
        Decodes the one-of-K code into class indices.

        :param onehot: (N, K) array containing one-of-K codings
        :return: (N,) array of class indices
        """
        return np.argmax(onehot, axis=1)  # type: ignore
