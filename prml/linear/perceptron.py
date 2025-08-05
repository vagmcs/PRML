# Dependencies
import numpy as np

from .classifier import Classifier


class Perceptron(Classifier):
    """
    Perceptron classifier.
    """

    def __init__(self) -> None:
        """
        Creates a perceptron classifier.
        """
        self._w: np.ndarray | None = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Trains the classifier.

        :param x: (N, D) array holding the input training data
        :param t: (N,) array holding the target classes
        """

        # initialize weights to zero
        self._w = np.zeros(np.size(x, 1))

        n_epochs = 1000
        for i in range(n_epochs):
            misclassified_indices = self.predict(x) != t
            x_error, t_error = x[misclassified_indices], t[misclassified_indices]
            idx = np.random.choice(len(x_error))
            self._w += x_error[idx] * t_error[idx]

            # if everything is correctly classified, then stop learning
            if all((x @ self._w) * t > 0):
                break

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction given an input.

        :param x: (N, D) array of samples to predict their output :return (N,) array
            holding the predicted classes
        """
        return np.sign(x @ self._w).astype(np.int32)  # type: ignore
