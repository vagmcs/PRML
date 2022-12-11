# Types
from typing import Callable, Tuple

# Dependencies
import numpy as np

# Project
from prml import datasets_dir


def generate_toy_data(
    f: Callable[[np.ndarray], np.ndarray], sample_size: int, std: float, domain: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a toy dataset given a function, a domain and a sample size. Then
    adds Gaussian noise to the samples having zero mean and the given standard deviation.

    :param f: a function
    :param sample_size: the size of the sample
    :param std: the standard deviation of the Gaussian noise
    :param domain: the domain range
    :return: a tuple of the input, target arrays
    """
    x = np.linspace(domain[0], domain[1], sample_size)
    t = f(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def load_old_faithful() -> np.ndarray:
    """
    Loads the old faithful dataset. Old Faithful is a hydrothermal geyser
    in Yellowstone National Park in the state of Wyoming, USA. The data
    comprises 272 observations, each representing a single eruption, and
    contains two variables corresponding to the duration of the eruption,
    and the time until the next eruption in minutes.

    :return: an array of shape (272, 2)
    """
    return np.genfromtxt(datasets_dir / "old_faithful.csv", dtype=float, delimiter=",", skip_header=1)

def load_planar_dataset(sample_size: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a binary class dataset having non-linear decision boundary.

    :param sample_size: the size of the sample
    :return: a tuple of the input, target arrays
    """
    # number of points per class
    n = int(sample_size / 2)

    # maximum ray of the flower
    a = 4

    x = np.zeros((sample_size, 2))
    y = np.zeros((sample_size, 1), dtype='uint8')

    for j in range(2):
        ix = range(n * j, n * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, n) + np.random.randn(n) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(n) * 0.2  # radius
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    x = x.T
    y = y.T

    return x, y
