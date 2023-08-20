# Dependencies
from typing import Optional
import numpy as np

# Project
from .modules import Module

# Find a small float to avoid division by zero
_epsilon = np.finfo(float).eps


def clip(x: np.ndarray) -> np.ndarray:
    return x.clip(min=_epsilon)


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
        _input = _input - np.max(_input, axis=1, keepdims=True)
        exps = np.exp(_input)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _backwards(self, _input: np.ndarray):
        """
        Backward pass of the softmax layer:

        https://binpord.github.io/2021/09/26/softmax_backprop.html

        :param _input: the backpropaged error
        :return: the derivative of the softmax
        """
        softmax = clip(self._forward(self._z))
        output = (_input - (_input * softmax).sum(axis=1, keepdims=True)) * softmax
        output[output == _epsilon] = 0
        return output