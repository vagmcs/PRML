# Types
from typing import Dict, Optional

# Standard Library
import abc

# Dependencies
import numpy as np


class Module(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self._training_mode = False

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
    def _forward(self, *inputs: np.ndarray, training_mode: bool = False) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _backwards(self, *inputs: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, *inputs: np.ndarray) -> np.ndarray:
        return self._forward(*inputs)


class LinearLayer(Module):
    def __init__(self, in_features: int, out_features: int, random_initialization: bool = True) -> None:
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._weights = (
            np.random.randn(out_features, in_features) * np.sqrt(1 / in_features)  # Xavier initialization
            if random_initialization
            else np.ones((out_features, in_features)) * 0.01
        )
        self._bias = (
            np.random.randn(out_features, 1) * np.sqrt(1 / in_features)
            if random_initialization
            else np.ones((out_features, 1)) * 0.01
        )
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

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._a = _input
        return self._weights @ self._a + self._bias

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        m = _input.shape[1]
        self._gradient["weights"] = (1 / m) * _input @ self._a.T
        self._gradient["bias"] = (1 / m) * np.sum(_input, axis=1, keepdims=True)
        return self._weights.T @ _input


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}.")
        self._p = p
        self._d = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._d = (np.random.rand(_input.shape[0], _input.shape[1]) < self._p).astype(float)
        return (self._d * _input) / self._p if training_mode else _input

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        return (self._d * _input) / self._p


class BatchNorm(Module):
    def __init__(self, momentum: float = 0.9, epsilon: float = 1e-5):
        self._momentum = momentum
        self._epsilon = epsilon
        self._running_mean = None
        self._running_var = None
        self._cache = None
        self._gamma = None
        self._beta = None
        self._gradient = {}

    @property
    def gamma(self) -> Optional[np.ndarray]:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: np.ndarray) -> None:
        self._gamma = gamma

    @property
    def beta(self) -> Optional[np.ndarray]:
        return self._beta

    @beta.setter
    def beta(self, beta: np.ndarray) -> None:
        self._beta = beta

    @property
    def gradient(self) -> Dict[str, np.ndarray]:
        return self._gradient

    def _forward(self, _inputs: np.ndarray, training_mode: bool = False) -> np.ndarray:
        D, _ = _inputs.shape

        self._gamma = np.ones((D, 1), dtype=_inputs.dtype) if self._gamma is None else self._gamma
        self._beta = np.zeros((D, 1), dtype=_inputs.dtype) if self._beta is None else self._beta

        running_mean = np.zeros((D, 1), dtype=_inputs.dtype) if self._running_mean is None else self._running_mean
        running_var = np.zeros((D, 1), dtype=_inputs.dtype) if self._running_var is None else self._running_var

        if training_mode:
            sample_mean = _inputs.mean(axis=1, keepdims=True)
            sample_var = _inputs.var(axis=1, keepdims=True)

            self._running_mean = self._momentum * running_mean + (1 - self._momentum) * sample_mean
            self._running_var = self._momentum * running_var + (1 - self._momentum) * sample_var

            _centered_inputs = _inputs - sample_mean
            _std = np.sqrt(sample_var + self._epsilon)
            _inputs_norm = _centered_inputs / _std

            self._cache = (_inputs_norm, _centered_inputs, _std, self._gamma)
        else:
            _inputs_norm = (_inputs - running_mean) / np.sqrt(running_var + self._epsilon)

        return self._gamma * _inputs_norm + self._beta

    def _backwards(self, _inputs: np.ndarray) -> np.ndarray:
        N = _inputs.shape[1]
        x_norm, x_centered, std, gamma = self._cache

        self._gradient["gamma"] = (_inputs * x_norm).sum(axis=1, keepdims=True)
        self._gradient["beta"] = _inputs.sum(axis=1, keepdims=True)

        dx_norm = _inputs * gamma
        dx_centered = dx_norm / std
        dmean = -(dx_centered.sum(axis=1, keepdims=True) + 2 / N * x_centered.sum(axis=1, keepdims=True))
        dstd = (dx_norm * x_centered * -(std ** (-2))).sum(axis=1, keepdims=True)
        dvar = dstd / 2 / std

        return dx_centered + (dmean + dvar * 2 * x_centered) / N


class Linear(Module):
    def __init__(self):
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return self._z

    def _backwards(self, _input: np.ndarray):
        return _input


class ReLU(Module):
    def __init__(self):
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return np.maximum(0, self._z)

    def _backwards(self, _input: np.ndarray):
        return _input * np.where(self._z > 0, 1, 0)


class TanH(Module):
    def __init__(self):
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return np.tanh(self._z)

    def _backwards(self, _input: np.ndarray):
        return _input * (1 - self._forward(self._z) ** 2)


class Sigmoid(Module):
    def __init__(self):
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return 1 / (1 + np.exp(-self._z))

    def _backwards(self, _input: np.ndarray):
        sigmoid = self._forward(self._z)
        return _input * sigmoid * (1 - sigmoid)
