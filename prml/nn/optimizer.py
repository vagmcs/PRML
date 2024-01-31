# Standard Library
import abc

# Dependencies
import numpy as np


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.iter = 0

    @abc.abstractmethod
    def _step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        pass

    def update(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self.iter += 1
        return self._step(parameters, gradient)


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.1, weight_decay: float = 0):
        super().__init__()
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def _step(self, parameters: np.ndarray, gradient: np.ndarray):
        return (
            parameters - self._learning_rate * gradient
            if self._weight_decay == 0
            else parameters - self._learning_rate * (gradient + self._weight_decay * parameters)
        )
