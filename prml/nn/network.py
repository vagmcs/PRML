# Futures
from __future__ import annotations

# Types
from typing import Iterator

# Dependencies
import numpy as np

# Project
from prml.nn.loss import Loss
from prml.nn.modules import Module
from prml.nn.optimizer import GradientDescent


class NeuralNetwork(Module):
    """
    Neural network base module.
    """

    def __init__(self, *modules: Module) -> None:
        self._modules = list(modules)
        self.optimizer = GradientDescent(self)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)

    def print_shape(self) -> None:
        for module in self:
            if module.weights is not None and module.bias is not None:
                print(f"Weights: {module.weights.shape}, bias: {module.bias.shape}")

    def _forward(self, _input: np.ndarray) -> np.ndarray:
        for module in self:
            _input = module(_input)
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
    ) -> None:
        if optimizer is not None:
            self.optimizer = optimizer

        if loss is None:
            raise ValueError("Loss function is not defined.")

        report_step = iterations / 10
        for i in range(iterations):
            y_hat = self(x)

            # Compute the cost
            cost = loss(y_hat, y)

            # Perform backpropagation
            self._backwards(loss.derivative(y_hat, y))

            # Perform optimization step
            self.optimizer.step()

            # Print the cost every 1000 iterations
            if verbose and i % report_step == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)
