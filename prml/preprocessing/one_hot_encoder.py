# Dependencies
import numpy as np


class OneHotEncoder(object):
    """
    One-encoder encoder.
    """

    @staticmethod
    def encode(class_indices: np.ndarray):
        """
        Encodes an array of class indices into one-of-k coding.

        :param class_indices: (N,) array of non-negative class indices
        :return: (N, K) array holding the one-of-K encodings
        """

        n_classes = np.max(class_indices) + 1
        return np.eye(n_classes)[class_indices]

    @staticmethod
    def decode(onehot: np.ndarray) -> np.ndarray:
        """
        Decodes the one-of-k code into class indices

        :param onehot: (N, K) array containing one-of-K codings
        :return: (N,) array of class indices
        """
        return np.argmax(onehot, axis=1)
