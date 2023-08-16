# Dependencies
from typing import Optional
import numpy as np

# Project
from .modules import Module


class Linear(Module):
    def __init__(self):
        self._z: Optional[np.ndarray] = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return self._z

    def _backwards(self, _input: np.ndarray):
        return _input


class ReLU(Module):
    def __init__(self):
        self._z: Optional[np.ndarray] = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return np.maximum(0, self._z)

    def _backwards(self, _input: np.ndarray):
        return _input * np.where(self._z > 0, 1, 0)


class TanH(Module):
    def __init__(self):
        self._z: Optional[np.ndarray] = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return np.tanh(self._z)

    def _backwards(self, _input: np.ndarray):
        return _input * (1 - self._forward(self._z) ** 2)


class Sigmoid(Module):
    def __init__(self):
        self._z: Optional[np.ndarray] = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return 1 / (1 + np.exp(-self._z))

    def _backwards(self, _input: np.ndarray):
        sigmoid = self._forward(self._z)
        return _input * sigmoid * (1 - sigmoid)


class Softmax(Module):
    def __init__(self) -> None:
        self._z: Optional[np.ndarray] = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        _input = _input - np.max(_input)
        exps = np.exp(_input)
        return exps / np.sum(exps)

    def _backwards(self, _input: np.ndarray):
        softmax = self._forward(self._z)
        return _input * softmax * (1 - softmax)
