# Standard Library
import abc

# Dependencies
import numpy as np

# Project
from .modules import Module

# Find a small float to avoid division by zero
_epsilon = np.finfo(float).eps


def clip(x: np.ndarray) -> np.ndarray:
    return x.clip(min=_epsilon)


class Loss(Module, metaclass=abc.ABCMeta):
    def derivative(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss function given an input and a target.

        :param _input: (N,) array of predictions
        :param _target: (N,) array of target values or classes
        :return: the derivative of the loss
        """
        return self._backwards(_input, _target)


class SSELoss(Loss):
    def _forward(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the sum-of-squares error

        E = 1/2 Σ(y - t)^2

        :param _input: (N,) array of predicted values
        :param _target: (N,) array of target values
        :return: the sum-of-squares error
        """
        return 0.5 * np.sum((_input - _target) ** 2)

    def _backwards(self, _input: np.ndarray, _target: np.ndarray):
        """
        Computes the derivative of the loss over the predicted values.

        dE/dy = y - t

        :param _input: (N,) array of predicted values
        :param _target: (N,) array of target values
        :return: the derivative of the loss
        """
        return _input - _target


class BinaryCrossEntropyLoss(Loss):
    def _forward(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the binary cross-entropy loss.

        E = -Σ tln(y) + (1-t)ln(1-y)

        :param _input: (N,) array of predicted values
        :param _target: (N,) array of target classes (0 or 1)
        :return: the binary cross-entropy loss
        """

        loss = np.log(clip(_input)) * _target + np.log(clip(1 - _input)) * (1 - _target)
        return -np.sum(loss) / _target.size

    def _backwards(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss over the predicted values.

        dE/dy = (1-t) / (1-y) - t / y

        :param _input: (N,) array of predicted values
        :param _target: (N,) array of target classes (0 or 1)
        :return: the derivative of the loss
        """
        return (1 - _target) / clip(1 - _input) - _target / clip(_input)


class CrossEntropyLoss(Loss):
    def _forward(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the cross-entropy loss.

        E = -Σ Σ tln(y)

        :param _input: (N, K) array of predicted values
        :param _target: (N, K) array of target classes
        :return: the cross-entropy loss
        """
        n, _ = _input.shape
        loss = np.log(clip(_input)) * _target
        return -np.sum(loss) / n

    def _backwards(self, _input: np.ndarray, _target: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss over the predicted values.

        dE/dy = -t / y

        :param _input: (N, K) array of predicted values
        :param _target: (N, K) array of target classes
        :return: the derivative of the loss
        """        
        return -_target / clip(_input)
