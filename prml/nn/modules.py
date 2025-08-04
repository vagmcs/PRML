# Futures
from __future__ import annotations

# Standard Library
import abc

# Dependencies
import numpy as np


class Module(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self._training_mode: bool = False

    @property
    def weights(self) -> np.ndarray | None:
        return None

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        pass

    @property
    def bias(self) -> np.ndarray | None:
        return None

    @bias.setter
    def bias(self, bias: np.ndarray) -> None:
        pass

    @property
    def gradient(self) -> dict[str, np.ndarray]:
        return {}

    @abc.abstractmethod
    def _forward(
        self, *inputs: np.ndarray, training_mode: bool = False, pertrubed_parameters: dict[str, np.array] = dict()
    ) -> np.ndarray:
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
            np.random.randn(out_features) * np.sqrt(1 / in_features)
            if random_initialization
            else np.ones(out_features) * 0.01
        )
        self._a: np.ndarray | None = None
        self._gradient = {}

    @property
    def weights(self) -> np.ndarray | None:
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self._weights = weights

    @property
    def bias(self) -> np.ndarray | None:
        return self._bias

    @bias.setter
    def bias(self, bias: np.ndarray) -> None:
        self._bias = bias

    @property
    def gradient(self) -> dict[str, np.ndarray]:
        return self._gradient

    def _forward(
        self, _input: np.ndarray, training_mode: bool = False, pertrubed_parameters: dict[str, np.array] = dict()
    ) -> np.ndarray:
        """
        Forward pass of the linear layer.

        y = a @ W.T + b

        :param _input: (N, D) array of training examples
        :param training_mode: enables training mode, defaults to False
        :return: (N, O) array of the linear transformation
        """
        self._a = _input

        if "weights" in pertrubed_parameters:
            return self._a @ pertrubed_parameters["weights"].T + self._bias
        elif "bias" in pertrubed_parameters:
            return self._a @ self._weights.T + pertrubed_parameters["bias"]

        return self._a @ self._weights.T + self._bias

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        """
        Backward pass of the linear layer.

        dE/dw =  1/m (δ^l @ a)

        :param _input:
        :return:
        """
        m, _ = _input.shape
        self._gradient["weights"] = (1 / m) * _input.T @ self._a
        self._gradient["bias"] = (1 / m) * np.sum(_input, axis=0, keepdims=True)
        return _input @ self._weights


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}.")
        self._p = p
        self._d = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._d = (np.random.rand(*_input.shape) < self._p).astype(float)
        return (self._d * _input) / self._p if training_mode else _input

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        return (self._d * _input) / self._p


class BatchNorm(Module):
    def __init__(self, momentum: float = 0.9, epsilon: float = 1e-5):
        super().__init__()
        self._momentum = momentum
        self._epsilon = epsilon
        self._running_mean = None
        self._running_var = None
        self._cache = None
        self._gamma = None
        self._beta = None
        self._gradient = {}

    @property
    def gamma(self) -> np.ndarray | None:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: np.ndarray) -> None:
        self._gamma = gamma

    @property
    def beta(self) -> np.ndarray | None:
        return self._beta

    @beta.setter
    def beta(self, beta: np.ndarray) -> None:
        self._beta = beta

    @property
    def gradient(self) -> dict[str, np.ndarray]:
        return self._gradient

    def _forward(self, _inputs: np.ndarray, training_mode: bool = False) -> np.ndarray:
        _, D = _inputs.shape

        self._gamma = np.ones((1, D), dtype=_inputs.dtype) if self._gamma is None else self._gamma
        self._beta = np.zeros((1, D), dtype=_inputs.dtype) if self._beta is None else self._beta

        running_mean = np.zeros((1, D), dtype=_inputs.dtype) if self._running_mean is None else self._running_mean
        running_var = np.zeros((1, D), dtype=_inputs.dtype) if self._running_var is None else self._running_var

        if training_mode:
            sample_mean = _inputs.mean(axis=0, keepdims=True)
            sample_var = _inputs.var(axis=0, keepdims=True)

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
        N, _ = _inputs.shape
        x_norm, x_centered, std, gamma = self._cache

        self._gradient["gamma"] = (_inputs * x_norm).sum(axis=0, keepdims=True)
        self._gradient["beta"] = _inputs.sum(axis=0, keepdims=True)

        dx_norm = _inputs * gamma
        dx_centered = dx_norm / std
        dmean = -(dx_centered.sum(axis=0, keepdims=True) + 2 / N * x_centered.sum(axis=1, keepdims=True))
        dstd = (dx_norm * x_centered * -(std ** (-2))).sum(axis=0, keepdims=True)
        dvar = dstd / 2 / std

        return dx_centered + (dmean + dvar * 2 * x_centered) / N


class Flatten(Module):
    def __init__(self) -> None:
        super().__init__()
        self._original_shape = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        """
        Flattens all dimensions of the input array except the first one that represents
        the number of training examples or mini-batch size.

        :param _input: (N, ...) input array
        :param training_mode: enables training mode, defaults to False
        :return: a flattened array
        """
        self._original_shape = _input.shape
        return _input.reshape(_input.shape[0], -1)

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        """
        Reverses the flattening operation.

        :param _input: the input from the previous layer
        :return: the (N, ...) un-flattened array
        """
        return _input.reshape(self._original_shape)


class Stack(Module):
    def __init__(self, module_groups: list[list[Module]]) -> None:
        super().__init__()
        self._module_groups = module_groups

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        component_outputs = []
        for module_group in self._module_groups:
            __input = _input
            for module in module_group:
                __input = module._forward(__input, training_mode)
            component_outputs.append(__input)
        return np.concatenate(component_outputs, axis=1)

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        component_outputs = []
        inputs = np.array_split(_input, len(self._module_groups), axis=1)
        for module_group, __input in zip(self._module_groups, inputs):
            for module in reversed(module_group):
                __input = module._backwards(__input)
            component_outputs.append(__input)
        return np.sum(component_outputs, axis=0)


class ConvLayer(Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: int = 1, padding: int = 0
    ) -> None:
        super().__init__()
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
    def weights(self) -> np.ndarray | None:
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self._weights = weights

    @property
    def bias(self) -> np.ndarray | None:
        return self._bias

    @bias.setter
    def bias(self, bias: np.ndarray) -> None:
        self._bias = bias

    @property
    def gradient(self) -> dict[str, np.ndarray]:
        return self._gradient

    def _forward(
        self, _inputs: np.ndarray, training_mode: bool = False, pertrubed_parameters: dict[str, np.array] = dict()
    ) -> np.ndarray:
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

                        if "weights" in pertrubed_parameters:
                            output[i, h, w, c] = np.sum(
                                input_slice * pertrubed_parameters["weights"][..., c] + self._bias[..., c]
                            )
                        elif "bias" in pertrubed_parameters:
                            output[i, h, w, c] = np.sum(
                                input_slice * self._weights[..., c] + pertrubed_parameters["bias"]
                            )
                        else:
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
                        ] += self._weights[:, :, :, c] * _input[i, h, w, c]

                        dW[:, :, :, c] += a_slice * _input[i, h, w, c]
                        db[:, :, :, c] += _input[i, h, w, c]

            if self._padding > 0:
                dA[i, :, :, :] = dA_padded[i, self._padding : -self._padding, self._padding : -self._padding, :]
            else:
                dA[i, :, :, :] = dA_padded[i, :, :, :]

        self._gradient["weights"] = dW
        self._gradient["bias"] = db

        return dA


class MaxPooling(Module):
    def __init__(self, pool_size: tuple[int, int], stride: int = 1) -> None:
        super().__init__()
        self._pool_size: tuple[int, int] = pool_size
        self._stride: int = stride
        self._a: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._a = _input
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

        return output

    def _backwards(self, _input: np.ndarray):
        m, n_height, n_width, channels = _input.shape
        dA = np.zeros(self._a.shape)

        for i in range(m):
            for h in range(n_height):
                for w in range(n_width):
                    for channel in range(channels):
                        a_slice = self._a[i, h : h + self._pool_size[0], w : w + self._pool_size[1], channel]
                        # find the index of the input slice (as an indicator matrix) that holds the maximum value
                        mask = a_slice == np.max(a_slice)
                        # mask the backward propagated loss using the same mask
                        dA[i, h : h + self._pool_size[0], w : w + self._pool_size[1], channel] += (
                            mask * _input[i, h, w, channel]
                        )

        return dA
