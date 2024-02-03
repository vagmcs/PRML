# Types
from typing import Any

# Standard Library
import abc

# Dependencies
import numpy as np


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def update(self, parameters: np.ndarray, gradient: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.1, weight_decay: float = 0) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def update(self, parameters: np.ndarray, gradient: np.ndarray, **kwargs: Any) -> np.ndarray:
        return parameters - self._learning_rate * (gradient + self._weight_decay * parameters)


class AdamW(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.1,
        weight_decay: float = 0,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._s = {}
        self._u = {}

    def update(self, parameters: np.ndarray, gradient: np.ndarray, **kwargs: Any) -> np.ndarray:
        if "parameters_name" not in kwargs:
            raise KeyError("Adam optimizer requires 'parameters' name for storing ")

        # compute and store the first and second moments of the gradients
        key = kwargs["parameters_name"]
        self._u[key] = self._beta_1 * self._u.setdefault(key, 0) + (1 - self._beta_1) * gradient
        u_hat = self._u[key] / (1 - self._beta_1)
        self._s[key] = self._beta_2 * self._s.setdefault(key, 0) + (1 - self._beta_2) * gradient**2
        s_hat = self._s[key] / (1 - self._beta_2)

        return parameters - self._learning_rate * (
            u_hat / np.sqrt(s_hat + self._epsilon) + self._weight_decay * parameters
        )
