# Futures
from __future__ import annotations

# Types
from typing import Iterator

# Dependencies
import numpy as np

# Project
from prml.nn.loss import Loss
from prml.nn.modules import Module
from prml.nn.optimizer import GradientDescent, Optimizer


class NeuralNetwork(Module):
    """
    Neural network base module.
    """

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self._modules = list(modules)
        self._optimizer = GradientDescent()

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)

    def print_shape(self) -> None:
        for module in self:
            if module.weights is not None and module.bias is not None:
                print(f"Weights: {module.weights.shape}, bias: {module.bias.shape}")

    def _check_gradients(self, x: np.ndarray, y: np.ndarray, loss: Loss, epsilon: float = 1e-4):
        layer_gradients = []
        for layer, module in enumerate(self):
            print(f"Layer {layer}")
            if module.weights is not None and module.bias is not None:
                weight_gradients = np.zeros(module.weights.shape)
                bias_gradients = np.zeros(module.bias.shape)

                # weight pertrubation
                for indices, _ in np.ndenumerate(module.weights):
                    weights_plus_e = np.copy(module.weights)
                    weights_minus_e = np.copy(module.weights)
                    weights_plus_e[indices] += epsilon
                    weights_minus_e[indices] -= epsilon

                    _input = x
                    for i, _module in enumerate(self):
                        _input = (
                            _module._forward(_input)
                            if i != layer
                            else _module._forward(_input, pertrubed_parameters={"weights": weights_plus_e})
                        )
                    cost_plus_e = loss(_input, y)

                    _input = x
                    for i, _module in enumerate(self):
                        _input = (
                            _module._forward(_input)
                            if i != layer
                            else _module._forward(_input, pertrubed_parameters={"weights": weights_minus_e})
                        )
                    cost_minus_e = loss(_input, y)

                    weight_gradients[indices] = (cost_plus_e - cost_minus_e) / (2 * epsilon)

                # bias pertrubation
                for indices, _ in np.ndenumerate(module.bias):
                    bias_plus_e = np.copy(module.bias)
                    bias_minus_e = np.copy(module.bias)
                    bias_plus_e[indices] += epsilon
                    bias_minus_e[indices] -= epsilon

                    _input = x
                    for i, _module in enumerate(self):
                        _input = (
                            _module._forward(_input)
                            if i != layer
                            else _module._forward(_input, pertrubed_parameters={"bias": bias_plus_e})
                        )
                    cost_plus_e = loss(_input, y)

                    _input = x
                    for i, _module in enumerate(self):
                        _input = (
                            _module._forward(_input)
                            if i != layer
                            else _module._forward(_input, pertrubed_parameters={"bias": bias_minus_e})
                        )
                    cost_minus_e = loss(_input, y)

                    bias_gradients[indices] = (cost_plus_e - cost_minus_e) / (2 * epsilon)

            layer_gradients.append((weight_gradients, bias_gradients))
        return layer_gradients

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
        epochs: int = 1000,
        loss: Loss | None = None,
        optimizer: Optimizer = None,
        batch_size: int | None = None,
        verbose: bool = True,
        report_steps: int = 10,
        check_gradients: bool = False,
        gradients_tol: float = 1e-6,
    ) -> None:
        if optimizer is not None:
            self._optimizer = optimizer

        if loss is None:
            raise ValueError("Loss function is not defined.")

        if batch_size is None:
            batch_size = len(x)

        indices = np.arange(0, len(x))
        report_step = epochs // report_steps

        for epoch in range(epochs):
            np.random.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                x_batch = x[indices[i : i + batch_size], :]
                y_batch = y[indices[i : i + batch_size], :]

                # Get predictions
                y_hat = self._forward(x_batch, True)

                # Compute the cost
                cost = loss(y_hat, y_batch)

                # Perform backpropagation
                self._backwards(loss.derivative(y_hat, y_batch))

                # Optionally perform gradient checking
                if check_gradients:
                    layer_gradients = self._check_gradients(x, y, loss)
                    for layer, module in enumerate(self._modules):
                        if module.weights is not None and module.bias is not None:
                            if (
                                np.linalg.norm(layer_gradients[layer][0] - module.gradient["weights"]) > gradients_tol
                                or np.linalg.norm(layer_gradients[layer][1] - module.gradient["bias"]) > gradients_tol
                            ):
                                raise RuntimeError(
                                    f"Gradient mismatch at layer {layer}:"
                                    f"\n\tWeights: {np.linalg.norm(layer_gradients[layer][0] - module.gradient['weights'])}"
                                    f"\n\tBias: {np.linalg.norm(layer_gradients[layer][1] - module.gradient['bias'])}"
                                )

                # Perform optimization step
                for layer, module in enumerate(self._modules):
                    if module.gradient:
                        if "weights" in module.gradient and "bias" in module.gradient:
                            module.weights = self._optimizer.update(
                                module.weights, module.gradient["weights"], parameters_name=f"weights_{layer}"
                            )
                            module.bias = self._optimizer.update(
                                module.bias, module.gradient["bias"], parameters_name=f"bias_{layer}"
                            )
                        elif "gamma" in module.gradient and "beta" in module.gradient:
                            module.gamma = self._optimizer.update(
                                module.gamma, module.gradient["gamma"], parameters_name=f"gamma_{layer}"
                            )
                            module.beta = self._optimizer.update(
                                module.beta, module.gradient["beta"], parameters_name=f"beta_{layer}"
                            )

            # Print the cost every 'report_step' iterations
            if verbose and epoch % report_step == 0:
                print(f"-- Epoch {epoch + 1} ---")
                print(f"Cost: {cost}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)
