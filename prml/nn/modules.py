# Types
from typing import Dict, Optional

# Standard Library
import abc

# Dependencies
import numpy as np


class Module(metaclass=abc.ABCMeta):
    @property
    def weights(self) -> Optional[np.ndarray]:
        return None

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        pass

    @property
    def bias(self) -> Optional[np.ndarray]:
        return None

    @bias.setter
    def bias(self, bias: np.ndarray) -> None:
        pass

    @property
    def gradient(self) -> Dict[str, np.ndarray]:
        return {}

    @abc.abstractmethod
    def _forward(self, *inputs: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _backwards(self, *inputs: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, *inputs: np.ndarray) -> np.ndarray:
        return self._forward(*inputs)


class LinearLayer(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        self._in_features = in_features
        self._out_features = out_features
        self._weights = np.random.randn(out_features, in_features) * 0.01
        self._bias = np.random.randn(out_features, 1) * 0.01
        self._a: np.ndarray | None = None
        self._gradient = {}

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self._weights = weights

    @property
    def bias(self) -> Optional[np.ndarray]:
        return self._bias

    @bias.setter
    def bias(self, bias: np.ndarray) -> None:
        self._bias = bias

    @property
    def gradient(self) -> Dict[str, np.ndarray]:
        return self._gradient

    def _forward(self, _input: np.ndarray) -> np.ndarray:
        self._a = _input
        return self._weights @ self._a + self._bias

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        m = _input.shape[1]
        self._gradient["weights"] = (1 / m) * _input @ self._a.T
        self._gradient["bias"] = (1 / m) * np.sum(_input, axis=1, keepdims=True)
        return self._weights.T @ _input


class Linear(Module):
    def __init__(self):
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray) -> np.ndarray:
        self._z = _input
        return self._z

    def _backwards(self, _input: np.ndarray):
        return _input


class ReLU(Module):
    def __init__(self):
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray) -> np.ndarray:
        self._z = _input
        return np.maximum(0, self._z)

    def _backwards(self, _input: np.ndarray):
        return _input * np.where(self._z > 0, 1, 0)


class TanH(Module):
    def __init__(self):
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray) -> np.ndarray:
        self._z = _input
        return np.tanh(self._z)

    def _backwards(self, _input: np.ndarray):
        return _input * (1 - self._forward(self._z) ** 2)


class Sigmoid(Module):
    def __init__(self):
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray) -> np.ndarray:
        self._z = _input
        return 1 / (1 + np.exp(-self._z))

    def _backwards(self, _input: np.ndarray):
        sigmoid = self._forward(self._z)
        return _input * sigmoid * (1 - sigmoid)
