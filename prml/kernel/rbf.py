# Dependencies
import numpy as np

# Project
from prml.kernel.kernel import Kernel


class RBF(Kernel):
    def __init__(self, theta: np.ndarray) -> None:
        super().__init__()
        self._theta = theta

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    @theta.setter
    def theta(self, theta: np.ndarray) -> None:
        self._theta = theta

    def __call__(self, x: np.ndarray, z: np.ndarray, pairwise: bool = True) -> np.ndarray:
        x = x[:, None] if x.ndim == 1 else x
        z = z[:, None] if z.ndim == 1 else z

        if pairwise:
            x = np.tile(x, (len(z), 1, 1)).transpose(1, 0, 2)
            z = np.tile(z, (len(x), 1, 1))

        return np.exp(-0.5 * np.sum(self._theta * (x - z) ** 2, axis=-1))

    def derivative(self, x: np.ndarray, z: np.ndarray, pairwise: bool = True) -> np.ndarray:
        x = x[:, None] if x.ndim == 1 else x
        z = z[:, None] if z.ndim == 1 else z

        if pairwise:
            x = np.tile(x, (len(z), 1, 1)).transpose(1, 0, 2)
            z = np.tile(z, (len(x), 1, 1))

        delta = np.exp(-0.5 * np.sum(self._theta * (x - z) ** 2, axis=-1))
        deltas = -0.5 * (x - z) ** 2 * delta[:, :, None]
        return deltas.T
