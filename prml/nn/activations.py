# Dependencies
import numpy as np

from .modules import Module

# Find a small float to avoid division by zero
_epsilon = np.finfo(float).eps


def clip(x: np.ndarray) -> np.ndarray:
    return x.clip(min=_epsilon)


class Linear(Module):
    def __init__(self):
        super().__init__()
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return self._z

    def _backwards(self, _input: np.ndarray):
        return _input


class Exp(Module):
    def __init__(self):
        super().__init__()
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return np.exp(self._z)

    def _backwards(self, _input: np.ndarray):
        return _input * np.exp(self._z)


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return np.maximum(0, self._z)

    def _backwards(self, _input: np.ndarray):
        """
        Note: Gradient checking on the ReLU function is known to have problems at x = 0. The ReLU function
        is defined as f(x) = max(0, x), where values less than 0 are clamped to 0, and values that are strictly
        positive retain the same. The problem encountered with gradient checking on ReLU is commonly known as
        the problem of kinks. Kinks refer to non-differentiable parts of an objective or activation function.
        For the ReLU function, the derivative approaching from the left of x = 0 and from the right of x = 0
        are not equal and so the derivative does not exist at x = 0 or more colloquially, there is a kink at x = 0.
        """
        return _input * np.where(self._z > 0, 1, 0)


class TanH(Module):
    def __init__(self):
        super().__init__()
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return np.tanh(self._z)

    def _backwards(self, _input: np.ndarray):
        return _input * (1 - self._forward(self._z) ** 2)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self._z: np.ndarray | None = None

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        self._z = _input
        return 1 / (1 + np.exp(-self._z))

    def _backwards(self, _input: np.ndarray):
        sigmoid = self._forward(self._z)
        return _input * sigmoid * (1 - sigmoid)


class Softmax(Module):
    def __init__(self) -> None:
        super().__init__()
        self._z: np.ndarray | None = None

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


class Concat(Module):
    def __init__(self, activations: list[Module]) -> None:
        super().__init__()
        self._activations = activations

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        component_outputs = []
        inputs = np.array_split(_input, len(self._activations), axis=1)
        for activation, __input in zip(self._activations, inputs):
            component_outputs.append(activation(__input))
        return np.concatenate(component_outputs, axis=1)

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        component_outputs = []
        inputs = np.array_split(_input, len(self._activations), axis=1)
        for activation, __input in zip(self._activations, inputs):
            component_outputs.append(activation._backwards(__input))
        return np.concatenate(component_outputs, axis=1)
