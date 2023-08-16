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


class ConvLayer(Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: int = 1, padding: int = 0
    ) -> None:
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._weights = np.random.randn(kernel_size[0], kernel_size[1], in_channels, out_channels) * np.sqrt(
            1 / in_channels
        )  # Xavier initialization
        self._bias = np.random.randn(1, 1, 1, out_channels) * np.sqrt(1 / in_channels)
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

    def _forward(self, _inputs: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._a = _inputs
        (m, height, width, channels) = _inputs.shape

        output_height = (height - self._kernel_size[0] + 2 * self._padding) // self._stride + 1
        output_width = (width - self._kernel_size[1] + 2 * self._padding) // self._stride + 1
        output = np.zeros((m, output_height, output_width, self._out_channels))

        # apply padding
        if self._padding > 0:
            padded_image = np.zeros((m, height + self._padding * 2, width + self._padding * 2, channels))
            padded_image[:, self._padding : -self._padding, self._padding : -self._padding, :] = _inputs
        else:
            padded_image = _inputs

        for i in range(m):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(self._out_channels):
                        input_slice = padded_image[
                            i,
                            h * self._stride : h * self._stride + self._kernel_size[0],
                            w * self._stride : w * self._stride + self._kernel_size[1],
                            :,
                        ]
                        output[i, h, w, c] = np.sum(input_slice * self._weights[..., c] + self._bias[..., c])

        return output

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        # dZ
        (_, a_height, a_width, a_channels) = self._a.shape
        (m, height, width, channels) = _input.shape

        dA = np.zeros(self._a.shape)
        dW = np.zeros(self._weights.shape)
        db = np.zeros(self._bias.shape)

        # apply padding
        if self._padding > 0:
            a_padded = np.zeros((m, a_height + self._padding * 2, a_width + self._padding * 2, a_channels))
            a_padded[:, self._padding : -self._padding, self._padding : -self._padding, :] = self._a
            dA_padded = np.zeros((m, a_height + self._padding * 2, a_width + self._padding * 2, a_channels))
            dA_padded[:, self._padding : -self._padding, self._padding : -self._padding, :] = dA
        else:
            a_padded = self._a
            dA_padded = dA

        for i in range(m):
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        a_slice = a_padded[
                            i,
                            h * self._stride : h * self._stride + self._kernel_size[0],
                            w * self._stride : w * self._stride + self._kernel_size[1],
                            :,
                        ]

                        dA_padded[
                            i,
                            h * self._stride : h * self._stride + self._kernel_size[0],
                            w * self._stride : w * self._stride + self._kernel_size[1],
                            :,
                        ] += (
                            self._weights[:, :, :, c] * _input[i, h, w, c]
                        )

                        dW[:, :, :, c] += a_slice * _input[i, h, w, c]
                        db[:, :, :, c] += _input[i, h, w, c]

            if self._padding > 0:
                dA[i, :, :, :] = dA_padded[i, self._padding : -self._padding, self._padding : -self._padding, :]
            else:
                dA[i, :, :, :] = dA_padded[i, :, :, :]

        self._gradient["weights"] = dW
        self._gradient["bias"] = db

        return dA


class Flatten(Module):
    def __init__(self) -> None:
        self._original_shape = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._original_shape = _input.shape
        return _input.reshape(_input.shape[0], -1).T

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        return _input.reshape(self._original_shape)


class MaxPooling(Module):
    def __init__(self, pool_size: tuple[int, int], stride: int = 1) -> None:
        self._pool_size: tuple[int, int] = pool_size
        self._stride: int = stride
        self._a: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        m, n_height, n_width, channels = _input.shape

        # shape of output
        output_height = (n_height - self._pool_size[0]) // self._stride + 1
        output_width = (n_width - self._pool_size[1]) // self._stride + 1
        output = np.zeros((m, output_height, output_width, channels))

        for i in range(m):
            for h in range(output_height):
                for w in range(output_width):
                    for channel in range(channels):
                        output[i, h, w, channel] = np.max(
                            _input[
                                i,
                                h * self._stride : h * self._stride + self._pool_size[0],
                                w * self._stride : w * self._stride + self._pool_size[1],
                                channel,
                            ]
                        )

        self._a = output
        return output

    def _backwards(self, _input: np.ndarray):
        m, n_height, n_width, channels = self._a.shape

        dA = np.zeros(self._a.shape)

        for i in range(m):
            for h in range(n_height):
                for w in range(n_width):
                    for channel in range(channels):
                        a_slice = self._a[i][h : h + self._pool_size[0], w : w + self._pool_size[1], channel]
                        # find the index of the input slice (as an indicator matrix) that holds the maximum value
                        mask = a_slice == np.max(a_slice)
                        # mask the backward propagated loss using the same mask
                        dA[i, h : h + self._pool_size[0], w : w + self._pool_size[1], channel] += (
                            mask * _input[i, h, w, channel]
                        )

        return dA
