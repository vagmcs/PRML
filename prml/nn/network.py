# Futures
from __future__ import annotations

# Types
from typing import Iterator

# Dependencies
import numpy as np

# Project
from prml.nn.loss import Loss
from prml.nn.modules import BatchNorm, ConvLayer, LinearLayer, Module
from prml.nn.optimizer import GradientDescent


class NeuralNetwork(Module):
    """
    Neural network base module.
    """

    def __init__(self, *modules: Module) -> None:
        self._modules = list(modules)
        self._optimizer = GradientDescent()

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)

    def print_shape(self) -> None:
        for module in self:
            if module.weights is not None and module.bias is not None:
                print(f"Weights: {module.weights.shape}, bias: {module.bias.shape}")

    def _forward(self, _input: np.ndarray, training_mode: bool = False) -> np.ndarray:
        for module in self:
            _input = module._forward(_input, training_mode)
        return _input

    def _backwards(self, _input: np.ndarray) -> np.ndarray:
        for module in reversed(self._modules):
            _input = module._backwards(_input)
        return _input

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        iterations: int = 10000,
        loss: Loss | None = None,
        optimizer=None,
        verbose: bool = True,
        report_steps: int = 10,
    ) -> None:
        if optimizer is not None:
            self._optimizer = optimizer

        if loss is None:
            raise ValueError("Loss function is not defined.")

        report_step = iterations / report_steps
        for i in range(iterations):
            y_hat = self._forward(x, True)

            # Compute the cost
            cost = loss(y_hat, y)

            # Perform backpropagation
            self._backwards(loss.derivative(y_hat, y))

            # Perform optimization step
            for module in self._modules:
                if module.gradient:
                    # linear layer
                    if isinstance(module, (LinearLayer, ConvLayer)):
                        module.weights = self._optimizer.update(module.weights, module.gradient["weights"])
                        module.bias = self._optimizer.update(module.bias, module.gradient["bias"])
                    # batch normalization layer
                    elif isinstance(module, BatchNorm):
                        module.gamma = self._optimizer.update(module.gamma, module.gradient["gamma"])
                        module.beta = self._optimizer.update(module.beta, module.gradient["beta"])

            # Print the cost every 1000 iterations
            if verbose and i % report_step == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)
