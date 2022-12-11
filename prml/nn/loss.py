# Standard Library
import abc

# Dependencies
import numpy as np

# Project
from .modules import Module

# Find a small float to avoid division by zero
_epsilon = np.finfo(float).eps

class Loss(Module, metaclass=abc.ABCMeta):
    def derivative(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        return self._backwards(_input, _target)


class SSELoss(Loss):
    def __init__(self):
        pass

    def _forward(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum((_input - _target) ** 2)

    def _backwards(self, _input: np.ndarray, _target: np.ndarray):
        return _input - _target


class CrossEntropyLoss(Loss):
    def __init__(self):
        pass

    def _forward(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        loss = np.log(_input.clip(_epsilon)) * _target + np.log((1 - _input).clip(_epsilon)) * (1 - _target)
        return -np.sum(loss) / _target.shape[1]

    def _backwards(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        return (1 - _target) / (1 - _input).clip(_epsilon) - (_target / _input.clip(_epsilon))
