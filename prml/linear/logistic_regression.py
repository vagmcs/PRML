# Dependencies
import numpy as np

# Project
from prml.linear.classifier import Classifier
from prml.preprocessing import OneHotEncoder


class LogisticRegression(Classifier):
    def __init__(self) -> None:
        self._w: np.ndarray | None = None

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit_lms(self, x: np.ndarray, t: np.ndarray, eta: float, n_iter: int = 1000) -> None:
        x = x[:, None] if x.ndim == 1 else x
        n, d = x.shape
        indices = np.arange(n)

        self._w = np.random.random(d)
        for _ in range(n_iter):
            # shuffle the data
            indices = np.random.permutation(indices)
            for i in indices:
                self._w = self._w - eta * (self._sigmoid(np.dot(self._w.T, x[i])) - t[i]) * x[i]

    def fit(self, x: np.ndarray, t: np.ndarray, n_iter: int = 1000) -> None:
        n, d = x.shape
        self._w = np.zeros(d)

        for _ in range(n_iter):
            prev_w = np.copy(self._w)

            y = self._sigmoid(x @ self._w)
            hessian = (x.T * y * (1 - y)) @ x
            self._w = self._w - np.linalg.inv(hessian) @ x.T @ (y - t)

            if np.allclose(self._w, prev_w):
                break

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x @ self._w)


class SoftmaxRegression(Classifier):
    def __init__(self) -> None:
        self._w: np.ndarray | None = None

    @staticmethod
    def _softmax(a: np.ndarray) -> np.ndarray:
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def fit(self, x: np.ndarray, t: np.ndarray, eta: float = 0.01, n_iter: int = 1000) -> None:
        x = x[:, None] if x.ndim == 1 else x
        n, d = x.shape
        indices = np.arange(n)
        t_one_hot = OneHotEncoder.encode(t)
        n_classes = t_one_hot.shape[1]

        self._w = np.random.random((d, n_classes))
        for _ in range(n_iter):
            # shuffle the data
            indices = np.random.permutation(indices)
            for i in indices:
                self._w = self._w - eta * (self._softmax(x[i] @ self._w) - t_one_hot[i]) * x[i, None].T

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self._softmax(x @ self._w), axis=1)
