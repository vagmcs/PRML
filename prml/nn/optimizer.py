# Standard Library
import abc

# Dependencies
import numpy as np

# Project
from .network import NeuralNetwork


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self, neural_net: NeuralNetwork):
        self.iter = 0
        self._neural_net = neural_net

    @abc.abstractmethod
    def update(self):
        pass

    def step(self) -> None:
        self.iter += 1
        self.update()


class GradientDescent(Optimizer):
    def __init__(self, neural_net: NeuralNetwork, learning_rate: float = 0.1):
        super().__init__(neural_net)
        self._learning_rate = learning_rate

    def update(self):
        for module in self._neural_net:
            if module.gradient:
                module.weights -= self._learning_rate * module.gradient["weights"]
                module.bias -= self._learning_rate * module.gradient["bias"]
