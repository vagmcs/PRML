# Standard Library
from abc import ABCMeta, abstractclassmethod

# Dependencies
import numpy as np


class Kernel(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def __call__(self, x: np.ndarray, z: np.ndarray, pairwise: bool) -> np.ndarray:
        pass

    @abstractclassmethod
    def derivative(self, x: np.ndarray, z: np.ndarray, pairwise: bool) -> np.ndarray:
        pass